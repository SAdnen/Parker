#define encoder0PinA  3
#define encoder0PinB  2

// Motor B Left Motor
#define dir1PinB   9 // direction control In1
#define dir2PinB  8 // direction control In2
#define speedPinB 10 // PWM speed control EnA

// Motor A Right Motor
#define dir1PinA   11 // direction control In1
#define dir2PinA  12 // direction control In2
#define speedPinA 13 // PWM speed control EnA

#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>

ros::NodeHandle nh;
std_msgs::Float32 msgA, msgB;
int Vl=0, Vr=0;

void rodomCallback(const std_msgs::Int32& r_cmd){
   Vr = r_cmd.data; 
}
void lodomCallback(const std_msgs::Int32& l_cmd){
   Vl = l_cmd.data; 
}
ros::Publisher rchatter("Right_Wheel_speed", &msgA);
ros::Publisher lchatter("Left_Wheel_speed", &msgB); 
ros::Subscriber<std_msgs::Int32> r_cmd("R_Commande", rodomCallback);
ros::Subscriber<std_msgs::Int32> l_cmd("L_Commande", lodomCallback);

volatile unsigned long encoderAPos=0;
volatile unsigned long encoderBPos=0;
unsigned long newAposition, newBposition;
unsigned long oldAposition = 0, oldBposition = 0;
unsigned long newtime;
unsigned long oldtime = 0;
int velA, velB;


void setup(){
  Serial.begin(19200);
  pinMode(dir1PinA,  OUTPUT);
  pinMode(dir2PinA,  OUTPUT);
  pinMode(speedPinA, OUTPUT);
  
  pinMode(dir1PinB,  OUTPUT);
  pinMode(dir2PinB,  OUTPUT);
  pinMode(speedPinB, OUTPUT);
  
  attachInterrupt(1, doEncoderA, CHANGE);
  attachInterrupt(0, doEncoderB, CHANGE);
  
  pinMode(encoder0PinA, INPUT_PULLUP);
  pinMode(encoder0PinB, INPUT_PULLUP);
  nh.initNode();
  nh.advertise(rchatter);
  nh.advertise(lchatter);
  nh.subscribe(r_cmd);
  nh.subscribe(l_cmd);
}

void loop(){
  newAposition = encoderAPos;
  newBposition = encoderBPos;
  newtime = millis();
  velA = (newAposition-oldAposition)*1000 /(newtime-oldtime);
  velB = (newBposition-oldBposition)*1000 /(newtime-oldtime);
  oldAposition = newAposition;
  oldBposition = newBposition;
  oldtime = newtime;
  
  Serial.print(velA);
  Serial.print(velB);
  msgA.data = velA;
  msgB.data = velB;


  
  rchatter.publish(&msgA);
  lchatter.publish(&msgB);
  
  digitalWrite(dir1PinA, HIGH);
  digitalWrite(dir2PinA, LOW);
  analogWrite(speedPinA, Vr);
  
  digitalWrite(dir1PinB, HIGH);
  digitalWrite(dir2PinB, LOW);
  analogWrite(speedPinB, Vl);
//  Serial.print('right commande, ');
//  Serial.println(Vr);
//  Serial.print('left commande, ');
//  Serial.println(Vl);
  
  nh.spinOnce();
  delay(100);
}

void doEncoderA(){encoderAPos++;}
void doEncoderB(){encoderBPos++;}
