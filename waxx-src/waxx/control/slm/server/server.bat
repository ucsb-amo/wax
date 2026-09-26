@echo off
setlocal
rem SLM server launcher.
rem
rem The Meadowlark SDK reaches the SLM as a second monitor. A Remote Desktop
rem session has only its own virtual display, so a server started there cannot
rem see the SLM and its LUT load fails. Run from Remote Desktop, this first
rem hands the session back to the PC's own screen (asks for admin; your Remote
rem Desktop window closes), then starts the server there. To look at it later,
rem use a tool that shows the real screen (VNC, TeamViewer, ...): reconnecting
rem with Remote Desktop takes the SLM away from the session again.

call :get_session
echo Session: %SESSION_NAME% (id %SESSION_ID%)
if /i not "%SESSION_NAME:~0,3%"=="rdp" goto wait_display

echo This is a Remote Desktop session: the SLM is not visible from here.
echo Handing the session to the PC's own screen. Accept the admin prompt;
echo your Remote Desktop window will close and the server starts on the PC.
powershell -NoProfile -Command "Start-Process tscon.exe -ArgumentList '%SESSION_ID%','/dest:console' -Verb RunAs"
if errorlevel 1 goto handback_failed

set /a tries=0
:wait_console
timeout /t 1 /nobreak >nul
call :get_session
if /i "%SESSION_NAME%"=="console" goto wait_display
set /a tries+=1
if %tries% lss 30 goto wait_console
goto handback_failed

:wait_display
rem Windows can take a few seconds to re-attach the monitors after a hand-back.
set /a tries=0
:wait_display_loop
call :count_screens
if %SCREENS% geq 2 goto start
set /a tries+=1
if %tries% geq 20 goto no_slm_display
timeout /t 1 /nobreak >nul
goto wait_display_loop

:start
echo Windows sees %SCREENS% displays. Starting the SLM server.
call %kpy%
cd /d "%~dp0"
python run_server.py
pause
exit /b

:handback_failed
echo The session is still not on the PC's own screen (%SESSION_NAME%). Not starting the server.
echo Start it at the PC, or from an Administrator prompt run: tscon %SESSION_ID% /dest:console
pause
exit /b 1

:no_slm_display
echo Windows sees only %SCREENS% display(s): the SLM is not showing up as a monitor.
echo Not starting the server, since it could not reach the SLM. Check the HDMI cable,
echo the controller LED (solid green = OK), and Settings - Display - Detect.
pause
exit /b 1

:get_session
rem The current session is the line of "query session" marked with ">".
set "SESSION_NAME="
set "SESSION_ID="
for /f "tokens=1,3 delims=> " %%a in ('query session 2^>nul ^| findstr /b ">"') do (
    set "SESSION_NAME=%%a"
    set "SESSION_ID=%%b"
)
if not defined SESSION_NAME set "SESSION_NAME=unknown"
exit /b

:count_screens
set "SCREENS=0"
for /f %%n in ('powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::AllScreens.Count"') do set "SCREENS=%%n"
exit /b
