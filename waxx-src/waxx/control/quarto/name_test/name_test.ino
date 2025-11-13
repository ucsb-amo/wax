#include "qCommand.h"
qCommand qC;
struct Cal 
{
    uint16_t cal_a;
    double cal_b;
    uint16_t cal_c;
    char cal_d[16];
};

void setup() {
  Serial.begin(115200);
  qC.addCommand("ping",ping);
}

void ping(qCommand& qC, Stream& S)
{
  struct Cal cal2;
  readNVMblock(&cal2, sizeof(cal2), 0xFA00);  
  Serial.println(cal2.cal_d); 
}

void loop()
{
  qC.readSerial(Serial);
}
