"""Scenes and watchdogs, run by the monitor server.

Both live on the server because it outlives the GUIs (closing a Device
Control window must not strand a coil that a scene was going to ramp down)
and the monitor experiment (which is killed whenever a run is submitted).

Everything advances in :meth:`OpRunner.tick`, called every ~0.2 s by the
server's runner thread (and directly by tests with a fake clock).  Ops are
submitted through the server's ordinary gate -- READY, registered, same
signature, no run pending -- and each one is re-checked by the monitor
against its hard limits, exactly as if a GUI had sent it.

Scenes (see :class:`~waxx.util.device_state.composite.Scene`): one at a
time; each step waits for the previous op's *result*.  Cancelling or a failed
step goes straight to the ``finally`` steps, which run best-effort (a failed
cleanup step does not skip the next).  If the monitor is gone, the cleanup
cannot run either, and the scene's result says so.

Watchdogs: a GUI arms one when a device becomes hazardous (e.g. a coil at
current) and it declares ``max_on_s``.  At the deadline every GUI is warned;
after ``grace_s`` more without "keep on" (``extend``), the server sends the
device's safe op itself (origin ``watchdog``).
"""

from __future__ import annotations

import itertools
import time
from typing import Callable

#: Give up on a step whose op has neither finished nor expired after this long
#: (the queue's own expiry normally answers first).
STEP_TIMEOUT_S = 180.0


class _Scene:
    def __init__(self, sid, request, client, operator):
        self.id = sid
        self.key = str(request.get("scene", ""))
        self.title = str(request.get("title", self.key))
        self.steps = list(request.get("steps") or [])
        self.finally_ = list(request.get("finally") or [])
        self.client = client
        self.operator = operator
        self.in_finally = False
        self.index = 0
        self.phase = "submit"
        self.seq = None
        self.t_step = 0.0
        self.hold_until = 0.0
        self.cancel = False
        self.failure = ""          # why the steps stopped early
        self.stopped_at = None     # 1-based step number where they stopped
        self.finally_errors = []

    @property
    def current_list(self):
        return self.finally_ if self.in_finally else self.steps

    def describe(self) -> dict:
        lst = self.current_list
        step = lst[self.index] if self.index < len(lst) else None
        return {"id": self.id, "scene": self.key, "title": self.title,
                "phase": "finally" if self.in_finally else "steps",
                "step": self.index + 1, "n": len(lst),
                "label": (step or {}).get("label", ""),
                "hold_left_s": None, "cancel": self.cancel}


class OpRunner:
    def __init__(self, submit: Callable[[dict, str], dict],
                 result: Callable[[int], dict | None],
                 broadcast: Callable[[dict], None],
                 journal=None, clock: Callable[[], float] = time.monotonic):
        self._submit = submit
        self._result = result
        self._broadcast = broadcast
        self._journal = journal
        self._clock = clock
        self._ids = itertools.count(1)
        self.scene: _Scene | None = None
        self.last_scene: dict | None = None
        self.watchdogs: dict[str, dict] = {}

    def _record(self, kind, **fields):
        if self._journal is not None:
            self._journal.record(kind, **fields)

    # --- scenes ----------------------------------------------------------------

    def start_scene(self, request: dict, client: str = "", operator: str = "") -> dict:
        if self.scene is not None:
            return {"status": "error",
                    "msg": f"scene '{self.scene.title}' is still running -- cancel it first"}
        steps = request.get("steps")
        if not isinstance(steps, list) or not steps:
            return {"status": "error", "msg": "scene has no steps"}
        for step in steps + list(request.get("finally") or []):
            if not isinstance(step, dict) or ("hold" not in step and "op" not in step):
                return {"status": "error", "msg": f"bad scene step {step!r}"}
        run = _Scene(next(self._ids), request, client, operator)
        self.scene = run
        self._record("scene_start", scene=run.key, id=run.id, client=client, operator=operator,
                     steps=[s.get("label") or s.get("op") for s in run.steps])
        self._emit(run, "running")
        self.tick()
        return {"status": "ok", "id": run.id}

    def cancel_scene(self, scene_id=None) -> dict:
        run = self.scene
        if run is None or (scene_id is not None and int(scene_id) != run.id):
            return {"status": "error", "msg": "no such scene running"}
        run.cancel = True
        self._record("scene_cancel", scene=run.key, id=run.id)
        self.tick()
        return {"status": "ok"}

    def _emit(self, run: _Scene, state: str, text: str = "") -> None:
        payload = {"type": "scene", "state": state, "text": text}
        payload.update(run.describe())
        if run.phase == "hold":
            payload["hold_left_s"] = max(run.hold_until - self._clock(), 0.)
        self._broadcast(payload)

    def _go_finally(self, run: _Scene, why: str = "") -> None:
        if not run.in_finally:
            if why:
                run.failure = why
                run.stopped_at = run.index + 1
            run.in_finally = True
            run.index = 0
            run.phase = "submit"

    def _finish(self, run: _Scene) -> None:
        if run.cancel:
            state = "cancelled"
            text = f"cancelled at step {run.stopped_at or len(run.steps)}/{len(run.steps)}"
        elif run.failure:
            state = "failed"
            text = f"failed at step {run.stopped_at}/{len(run.steps)}: {run.failure}"
        else:
            state, text = "done", "done"
        if run.finally_:
            if run.finally_errors:
                text += "; cleanup had problems: " + "; ".join(run.finally_errors)
                if state == "done":
                    state = "failed"
            else:
                text += "; cleanup ran"
        self._emit(run, state, text)
        self._record("scene_end", scene=run.key, id=run.id, state=state, text=text)
        self.last_scene = {"id": run.id, "scene": run.key, "title": run.title,
                           "state": state, "text": text}
        self.scene = None

    def _advance(self, run: _Scene) -> None:
        run.index += 1
        run.phase = "submit"
        run.seq = None
        if run.index >= len(run.current_list):
            if run.in_finally:
                self._finish(run)
            else:
                self._go_finally(run)
                if not run.finally_:
                    self._finish(run)

    def _tick_scene(self) -> None:
        # A few transitions per tick at most, so a scene with no holds still
        # takes one tick per op (each waits for the monitor anyway).
        for _ in range(8):
            run = self.scene
            if run is None:
                return
            now = self._clock()
            if run.cancel and not run.in_finally and run.phase != "wait":
                # (an op in flight is let finish -- its events are already
                # on the timeline -- and the cleanup starts after it)
                run.stopped_at = run.stopped_at or run.index + 1
                self._go_finally(run)
                continue
            lst = run.current_list
            if run.index >= len(lst):
                self._advance(run)
                continue
            step = lst[run.index]
            if run.phase == "submit":
                if "hold" in step:
                    run.phase = "hold"
                    run.hold_until = now + max(float(step["hold"]), 0.)
                    self._emit(run, "running")
                    return
                reply = self._submit({"op": step.get("op"), "sig": step.get("sig"),
                                      "args": step.get("args") or {},
                                      "payload": step.get("payload") or {},
                                      "client": run.client, "operator": run.operator},
                                     "finally" if run.in_finally else "scene")
                if reply.get("status") != "ok":
                    msg = str(reply.get("msg", "refused"))
                    if run.in_finally:
                        run.finally_errors.append(f"{step.get('label') or step.get('op')}: "
                                                  f"not run ({msg})")
                        self._advance(run)
                        continue
                    self._go_finally(run, f"{step.get('label') or step.get('op')}: {msg}")
                    continue
                run.seq = int(reply["seq"])
                run.t_step = now
                run.phase = "wait"
                self._emit(run, "running")
                return
            if run.phase == "hold":
                if now >= run.hold_until:
                    self._advance(run)
                    continue
                return
            if run.phase == "wait":
                result = self._result(run.seq)
                if result is None:
                    if now - run.t_step > STEP_TIMEOUT_S:
                        result = {"ok": False, "text": f"no result after {STEP_TIMEOUT_S:.0f} s"}
                    else:
                        return
                label = step.get("label") or step.get("op")
                if result.get("ok"):
                    self._advance(run)
                elif run.in_finally:
                    run.finally_errors.append(f"{label}: {result.get('text', 'failed')}")
                    self._advance(run)
                else:
                    self._go_finally(run, f"{label}: {result.get('text', 'failed')}")
                continue

    # --- watchdogs -----------------------------------------------------------------

    def arm_watchdog(self, request: dict) -> dict:
        device = str(request.get("device", ""))
        try:
            max_on = float(request["max_on_s"])
            grace = float(request.get("grace_s", 120.))
        except (KeyError, TypeError, ValueError):
            return {"status": "error", "msg": "watchdog needs max_on_s"}
        if not device or not request.get("op") or not request.get("sig"):
            return {"status": "error", "msg": "watchdog needs device, op and sig"}
        if device in self.watchdogs:
            return {"status": "ok", "armed": True, "already": True,
                    "fires_in_s": self._fires_in(self.watchdogs[device])}
        now = self._clock()
        self.watchdogs[device] = {
            "device": device, "title": str(request.get("title", device)),
            "deadline": now + max_on, "max_on_s": max_on, "grace_s": grace,
            "op": request["op"], "sig": request["sig"], "args": dict(request.get("args") or {}),
            "client": str(request.get("client", "")),
            "operator": str(request.get("operator", "")), "warned": False,
        }
        self._record("watchdog_arm", device=device, max_on_s=max_on, grace_s=grace,
                     client=request.get("client"))
        return {"status": "ok", "armed": True, "fires_in_s": max_on + grace}

    def extend_watchdog(self, device: str, operator: str = "") -> dict:
        dog = self.watchdogs.get(device)
        if dog is None:
            return {"status": "error", "msg": "no watchdog armed for that device"}
        dog["deadline"] = self._clock() + dog["max_on_s"]
        dog["warned"] = False
        self._record("watchdog_extend", device=device, operator=operator)
        self._broadcast({"type": "watchdog", "device": device, "state": "extended",
                         "fires_in_s": self._fires_in(dog)})
        return {"status": "ok", "fires_in_s": self._fires_in(dog)}

    def disarm_watchdog(self, device: str) -> dict:
        if self.watchdogs.pop(device, None) is not None:
            self._record("watchdog_disarm", device=device)
            self._broadcast({"type": "watchdog", "device": device, "state": "disarmed"})
        return {"status": "ok"}

    def _fires_in(self, dog) -> float:
        return max(dog["deadline"] + dog["grace_s"] - self._clock(), 0.)

    def _tick_watchdogs(self) -> None:
        now = self._clock()
        for device, dog in list(self.watchdogs.items()):
            if now >= dog["deadline"] and not dog["warned"]:
                dog["warned"] = True
                self._broadcast({"type": "watchdog", "device": device, "state": "warning",
                                 "title": dog["title"], "fires_in_s": self._fires_in(dog)})
                self._record("watchdog_warning", device=device)
            if now >= dog["deadline"] + dog["grace_s"]:
                del self.watchdogs[device]
                reply = self._submit({"op": dog["op"], "sig": dog["sig"], "args": dog["args"],
                                      "payload": {}, "client": "watchdog",
                                      "operator": dog["operator"]}, "watchdog")
                ok = reply.get("status") == "ok"
                text = (f"sent {dog['op']} (#{reply.get('seq')})" if ok
                        else f"could NOT send {dog['op']}: {reply.get('msg')}")
                self._broadcast({"type": "watchdog", "device": device, "title": dog["title"],
                                 "state": "fired" if ok else "failed", "text": text})
                self._record("watchdog_fired", device=device, ok=ok, text=text)

    # --- all ---------------------------------------------------------------------

    def tick(self) -> None:
        self._tick_scene()
        self._tick_watchdogs()

    def info(self) -> dict:
        run = self.scene
        scene = None
        if run is not None:
            scene = run.describe()
            if run.phase == "hold":
                scene["hold_left_s"] = max(run.hold_until - self._clock(), 0.)
        return {"scene": scene, "last_scene": self.last_scene,
                "watchdogs": {d: {"title": w["title"], "fires_in_s": self._fires_in(w),
                                  "warned": w["warned"]}
                              for d, w in self.watchdogs.items()}}

    def abandon(self, reason: str) -> None:
        """Server shutting down or monitor gone for good: stop cleanly."""
        if self.scene is not None:
            self.scene.cancel = True
            self._go_finally(self.scene, reason)
