#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#define img1_x 50
#define img1_y 50
#define img2_x 500
#define img2_y 50

using namespace cv;
using namespace std;

Mat image1, image2, morph_button;
Mat OutputImage = Mat::zeros(Size(800, 300), CV_8UC3);
Point image1_start, image1_end, image2_start, image2_end;
vector<Point> P, Q, P_prime, Q_prime;
bool img1_start_check = false, img1_end_check = false;
bool img2_start_check = false, img2_end_check = false, morphing_check = false;

Point perp(Point p) {
	Point trans(p.y, -p.x);
	return trans;
}

void morphing() {
	Mat map1x[3], map1y[3], map2x[3], map2y[3];
	Mat mix[3], LR[3], RL[3];
	Point  X, X_prime, XSum21, XSum12;
	double u, v, weight, WeightSum21, WeightSum12, dist;
	int arrow_num = min(P.size(), P_prime.size());

	for (int i = 0; i < 3; i++)
	{
		map1x[i].create(image1.size(), CV_32FC1);
		map1y[i].create(image1.size(), CV_32FC1);
		map2x[i].create(image1.size(), CV_32FC1);
		map2y[i].create(image1.size(), CV_32FC1);
		LR[i].create(image1.size(), CV_32FC1);
		RL[i].create(image1.size(), CV_32FC1);
		mix[i].create(image1.size(), CV_32FC1);
	}

	for (int x = 0; x < image1.cols; x++) {
		for (int y = 0; y < image1.rows; y++) {
			X = Point(x, y);
			XSum21 = Point(0, 0);
			XSum12 = Point(0, 0);
			WeightSum21 = 0.0;
			WeightSum12 = 0.0;
			
			for (int z = 0; z < arrow_num; z++) {
				// 2->1
				// u  = (X-P)*(Q-P) / ||Q-P||^2
				u = (((Q[z] - P[z]).x * (X - P[z]).x + (Q[z] - P[z]).y * (X - P[z]).y)) /
					(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2));

				// v = (X-P)*Perpendicular(Q-P) / ||Q-P||
				v = (((X - P[z]).x * (Q[z] - P[z]).y - (X - P[z]).y * (Q[z] - P[z]).x)) /
					sqrt(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2));

				// X' = P' + u*(Q'-P') + v*Perpendicular(Q'-P') / || Q'-P'||
				X_prime = P_prime[z] + u * (Q_prime[z] - P_prime[z]) + v * perp(Q_prime[z] - P_prime[z]) /
					sqrt(pow((Q_prime[z] - P_prime[z]).x, 2) + pow((Q_prime[z] - P_prime[z]).y, 2));
				// weight[z] = [ ( length[z]^p ) / ( a + dist[z] ) ] ^b
				// a = 1 , b = 2 , p = 0 
				if (u < 0)
					dist = sqrt(pow((X_prime - P_prime[z]).x, 2) + pow((X_prime - P_prime[z]).y, 2));
				else if (u > 1)
					dist = sqrt(pow((X_prime - Q_prime[z]).x, 2) + pow((X_prime - Q_prime[z]).y, 2));
				else
					dist = abs(v);
				weight = pow((pow(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2), 0.5) / (1 + dist)), 2);
				// XSum = Xsum + X'[z]*weight
				XSum21 = XSum21 + X_prime * weight;
				// WeightSum += weight[z]
				WeightSum21 = WeightSum21 + weight;

				// 1->2
				// u  = (X-P')*(Q'-P') / ||Q'-P'||^2
				u = ((X - P_prime[z]).x*(Q_prime[z] - P_prime[z]).x + (X - P_prime[z]).y*(Q_prime[z] - P_prime[z]).y) /
					(pow((Q_prime[z] - P_prime[z]).x, 2) + pow((Q_prime[z] - P_prime[z]).y, 2));

				// v = (X-P')*Perpendicular(Q'-P') / ||Q'-P'||
				v = ((X - P_prime[z]).x*(Q_prime[z] - P_prime[z]).y - (X - P_prime[z]).y*(Q_prime[z] - P_prime[z]).x) /
					sqrt((pow((Q_prime[z] - P_prime[z]).x, 2) + pow((Q_prime[z] - P_prime[z]).y, 2)));

				// X' = P + u*(Q-P) + v*Perpendicular(Q-P) / || Q-P||
				X_prime = P[z] + u * (Q[z] - P[z]) + v * perp(Q[z]-P[z])/sqrt(pow((Q[z] - P[z]).x,2) + pow((Q[z] - P[z]).y,2));

				// weight[z] = [ ( length[z]^p ) / ( a + dist[z] ) ] ^b
				// a = 1 , b = 2 , p = 0 
				if (u < 0)
					dist = sqrt(pow((X_prime - P_prime[z]).x, 2) + pow((X_prime - P_prime[z]).y, 2));
				else if (u > 1)
					dist = sqrt(pow((X_prime - Q_prime[z]).x, 2) + pow((X_prime - Q_prime[z]).y, 2));
				else
					dist = abs(v);
				weight = pow((pow(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2), 0.5) / (1 + dist)), 2);

				// XSum = Xsum + X'[z]*weight
				XSum12 = XSum12 + X_prime * weight;
				// WeightSum += weight[z]
				WeightSum12 = WeightSum12 + weight;
			}
			for (int k = 0; k < 3; k++)
			{
				map1x[k].at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*(0.75 - 0.25*k) + X*(0.25 + 0.25*k)).x;
				map1y[k].at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*(0.75 - 0.25*k) + X*(0.25 + 0.25*k)).y;
				map2x[k].at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*(0.25 + 0.25*k) + X*(0.75 - 0.25*k)).x;
				map2y[k].at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*(0.25 + 0.25*k) + X*(0.75 - 0.25*k)).y;
			}
		}
	}
	for (int i = 0; i < 3; i++)
	{
		cv::remap(image2, RL[i], map1x[i], map1y[i], INTER_LINEAR);
		cv::remap(image1, LR[i], map2x[i], map2y[i], INTER_LINEAR);
		cv::addWeighted(LR[i], (0.75-0.25*i), RL[i], (0.25+0.25*i), 0.0, mix[i]);
	}

	cv::namedWindow("t = 0");
	cv::namedWindow("t = 0.25");
	cv::namedWindow("t = 0.5");
	cv::namedWindow("t = 0.75");
	cv::namedWindow("t = 1");
	
	cv::imshow("t = 0", image1);
	cv::imshow("t = 0.25", mix[0]);
	cv::imshow("t = 0.5", mix[1]);
	cv::imshow("t = 0.75", mix[2]);
	cv::imshow("t = 1", image2);

	cv::imwrite("t025.jpg", mix[0]);
	cv::imwrite("t050.jpg", mix[1]);
	cv::imwrite("t075.jpg", mix[2]);
}

void mouse_event(int event, int x, int y, int flags, void* userdata)
{
	Point image1_original(img1_x, img1_y), image2_original(img2_x, img2_y);
	if (morphing_check == true)
	{
		morphing_check = false;
		morphing();
	}
	// image1
	if (x >= img1_x && x <= (img1_x+image1.cols) && y >= img1_y && y <= (img1_y+image1.rows))
	{
		if (event == EVENT_LBUTTONDOWN)
		{
			std::cout << "image1" << endl;
			std::cout << "start at (" << x - image1_original.x << ", " << y - image1_original.y << ")" << endl;
			image1_start = Point(x, y);
			img1_start_check = true;
		}
		else if (event == EVENT_LBUTTONUP)
		{
			std::cout << "end at (" << x - image1_original.x << ", " << y - image1_original.y << ")" << endl;
			image1_end = Point(x, y);
			img1_end_check = true;
			if (img1_start_check == true && img1_end_check == true) {
				img1_start_check = false;
				img1_end_check = false;
				arrowedLine(OutputImage, image1_start, image1_end, Scalar(0, 255, 255), 3);
				Point tmpP = image1_start - image1_original;
				Point tmpQ = image1_end - image1_original;
				P.push_back(tmpP);
				Q.push_back(tmpQ);
			}
		}
	}
	// image2 
	else if (x >= img2_x && x <= (img2_x+image2.cols) && y >= img2_y && y <= (img2_y+image2.rows))
	{
		if (event == EVENT_LBUTTONDOWN)
		{
			std::cout << "image2" << endl;
			std::cout << "start at (" << x - image2_original.x << ", " << y - image2_original.y << ")" << endl;
			image2_start = Point(x, y);
			img2_start_check = true;
		}
		else if (event == EVENT_LBUTTONUP)
		{
			std::cout << "end at (" << x - image2_original.x << ", " << y - image2_original.y << ")" << endl;
			image2_end = Point(x, y);
			img2_end_check = true;

			if (img2_start_check == true && img2_end_check == true) {
				img2_start_check = false;
				img2_end_check = false;
				arrowedLine(OutputImage, image2_start, image2_end, Scalar(0, 255, 255), 3);
				Point tmpP2 = image2_start - image2_original;
				Point tmpQ2 = image2_end - image2_original;
				P_prime.push_back(tmpP2);
				Q_prime.push_back(tmpQ2);
			}
		}
	}
	else if (x >= 325 && x < 485 && y >= 70 && y <= 150)
	{
		if (event == EVENT_LBUTTONDOWN)
			morphing_check = true;
	}
	else {
		if (event == EVENT_LBUTTONDOWN)
			cout << "click the invalid area" << endl;
	}
	cv::imshow("Morphing", OutputImage);
}

int main(int argc, char** argv)
{
	image1 = imread("women.jpg");
	image2 = imread("cheetah.jpg");
	morph_button = imread("morph_button.png");
	OutputImage.setTo(Scalar(100, 0, 0));
	if (!image1.data || !image2.data || !morph_button.data)
	{
		std::cout << "load image error" << endl;
		exit(-1);
	}
	Rect R1(img1_x, img1_y, image1.cols, image1.rows);
	Rect R2(img2_x, img2_y, image2.cols, image2.rows);
	Rect button_R(325, 70, 160, 80);
	image1.copyTo(OutputImage(R1));
	image2.copyTo(OutputImage(R2));
	resize(morph_button, morph_button, Size(160, 80));
	morph_button.copyTo(OutputImage(button_R));
	std::cout << "img1 row: " << image1.rows << " col: " << image1.cols << endl;
	std::cout << "img2 row: " << image2.rows << " col: " << image2.cols << endl;
	cv::namedWindow("Morphing");
	cvSetMouseCallback("Morphing", mouse_event,NULL);
	cv::imshow("Morphing", OutputImage);
	cv::waitKey(0);
	cv::destroyWindow("Morphing");
	return 0;
}