/*
 * controller.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: adnen
 */
#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Int32.h"
#include <sstream>
const double V_CONS_RPM = 80;
// PID controller coefficients
const double Kp_r = 0.5;
const double Ki_r = 0.2;
const double Kd_r = 0;

const double Kp_l = 0.5;
const double Ki_l = 0.2;
const double Kd_l = 0;
std_msgs::Float32 wheel_speed;
std_msgs::Int32 pwm_r,
				pwm_l;

float vr_cons, vr_now, er, er_prev, I_er = 0, D_er = 0,
	  vl_cons, vl_now, el, el_prev, I_el = 0, D_el = 0;
int pwm_int_r,
	pwm_int_l;

double sp_r, si_r, sd_r, dt,
	   sp_l, si_l, sd_l;

void rOdomCallback(const std_msgs::Float32ConstPtr& odom){
	vr_now = odom->data;
	ROS_INFO("Right Wheel Odom = [%f]", vr_now);
}

void lOdomCallback(const std_msgs::Float32ConstPtr& odom){
	vl_now = odom->data;
	ROS_INFO("Left Wheel Odom = [%f]", vl_now);
}

int main (int argc, char **argv){

	std_msgs::Float32 msg;

	ros::init(argc, argv,"Controller");
	ros::NodeHandle node;
	ros::Publisher r_cmd = node.advertise<std_msgs::Int32>("R_Commande", 1);
	ros::Publisher l_cmd = node.advertise<std_msgs::Int32>("L_Commande", 1);
	ros::Subscriber r_odom = node.subscribe("Right_Wheel_speed", 1, rOdomCallback);
	ros::Subscriber l_odom = node.subscribe("Left_Wheel_speed", 1, lOdomCallback);
	ros::Rate rate(10);
	ros::Time newtime, oldtime;

	newtime = ros::Time::now();
	oldtime = ros::Time::now();

	vr_cons = V_CONS_RPM;
	vl_cons = V_CONS_RPM;
	er_prev = er;
	el_prev = el;

	while(ros::ok()){
		//ROS_INFO("Wheel Speed =  [%f]", wheel_speed);
		newtime = ros::Time::now();
		dt = (newtime - oldtime).toSec();
/*===========================Right Wheel PID ====================================================*/
		er = vr_cons - vr_now; //computing right wheel velocity error
		ROS_INFO("Right Wheel Error  =[%f]", er);
		sp_r = Kp_r * er; //output of Proportional controller for right wheel

		I_er = I_er + er * dt; //computing integral of right wheel velocity error
		si_r = Ki_r * I_er; //output of Integral controller for right wheel

		D_er = (er - er_prev)/dt; // computing derivative of error for the right wheel
		sd_r = Kd_r * D_er; // output  of Derivative controller for the right wheel

		pwm_int_r = ceil(((sp_r+si_r) + 160) / 2.57); // computing the power card input (pwm) using a linear-regression curve
		pwm_r.data = pwm_int_r;
		ROS_INFO("right pwm =[%d]", pwm_r.data);

		//r_cmd.publish(pwm_r);
		if(pwm_int_r < 250 and pwm_int_r > 0){ // for security reasons we have to control pwm command to avoid serious damages
		r_cmd.publish(pwm_r);
		}
/*================================Left Wheel PID=================================================*/
		el = vl_cons - vl_now; //computing left wheel velocity error
		ROS_INFO("Left Wheel Error   =[%f]", el);
		sp_l = Kp_l * el; //output of Proportional controller for left wheel

		I_el = I_el + el * dt; //computing integral of left wheel velocity error
		si_l = Ki_l * I_el; //output of Integral controller fir left wheel

		D_el = (el - el_prev)/dt; // computing derivative of the error for the left wheel
		sd_l = Kd_l * D_el; // output  of Derivative controller for the left wheel

		pwm_int_l = ceil(((sp_l+si_l) + 160) / 2.57); // computing the power card input (pwm) using a linear-regression curve
		pwm_l.data = pwm_int_l;
		ROS_INFO("left pwm  =[%d]", pwm_l.data);
		//l_cmd.publish(pwm_l);
		if(pwm_int_l < 250 and pwm_int_l > 0){ // for security reasons we have to control pwm command to avoid serious damages
		l_cmd.publish(pwm_l);
		}
/*==================================================================================================*/
		rate.sleep();
		ros::spinOnce();
		oldtime = newtime;
		er_prev = er;
		el_prev = el;
	}

	return 0;

}



