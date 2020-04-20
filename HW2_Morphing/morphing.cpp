#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define img1_x 50
#define img1_y 50
#define img2_x 500
#define img2_y 50

using namespace cv;
using namespace std;

Mat OutputImage = Mat::zeros(Size(800, 300), CV_8UC3);
Mat image1, image2, morph_button;
Mat result14, result24, result34;

Point image1_start, image1_end, image2_start, image2_end;
Point image1_original (img1_x,img1_y), image2_original(img2_x, img2_y);
Point P[50], Q[50], P_prime[50], Q_prime[50];

bool img1_start_check = false, img1_end_check = false;
bool img2_start_check = false, img2_end_check = false, button_true = false;

int image1_arrow_index = 0, image2_arrow_index = 0;

Point perp(Point p) {
	Point transform(p.y, -p.x);
	return transform;
}

void morphing() {
	Mat one2two_14, one2two_24, one2two_34;
	Mat two2one_14, two2one_24, two2one_34;
	Mat mix_14, mix_24, mix_34;

	Mat mapx_14, mapy_14;
	Mat mapx_24, mapy_24;
	Mat mapx_34, mapy_34;

	Mat mapx_14_2, mapy_14_2;
	Mat mapx_24_2, mapy_24_2;
	Mat mapx_34_2, mapy_34_2;

	Point  X, X_prime, XSum21, XSum12;
	double u, v, weight, WeightSum21, WeightSum12, dist;
	int arrow_num = min(image1_arrow_index, image2_arrow_index);

	one2two_14.create(image1.size(), CV_32FC1);
	one2two_24.create(image1.size(), CV_32FC1);
	one2two_34.create(image1.size(), CV_32FC1);
	two2one_14.create(image1.size(), CV_32FC1);
	two2one_24.create(image1.size(), CV_32FC1);
	two2one_34.create(image1.size(), CV_32FC1);

	mix_14.create(image1.size(), CV_32FC1);
	mix_24.create(image1.size(), CV_32FC1);
	mix_34.create(image1.size(), CV_32FC1);

	mapx_14.create(image1.size(), CV_32FC1);
	mapy_14.create(image1.size(), CV_32FC1);
	mapx_24.create(image1.size(), CV_32FC1);
	mapy_24.create(image1.size(), CV_32FC1);
	mapx_34.create(image1.size(), CV_32FC1);
	mapy_34.create(image1.size(), CV_32FC1);

	mapx_14_2.create(image1.size(), CV_32FC1);
	mapy_14_2.create(image1.size(), CV_32FC1);
	mapx_24_2.create(image1.size(), CV_32FC1);
	mapy_24_2.create(image1.size(), CV_32FC1);
	mapx_34_2.create(image1.size(), CV_32FC1);
	mapy_34_2.create(image1.size(), CV_32FC1);

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
					(pow((((Q[z] - P[z]).x)), 2) + pow((((Q[z] - P[z]).y)), 2));

				// v = (X-P)*Perpendicular(Q-P) / ||Q-P||
				v = (((X - P[z]).x * (Q[z] - P[z]).y - (X - P[z]).y * (Q[z] - P[z]).x)) /
					(pow((pow((((Q[z] - P[z]).x)), 2) + pow((((Q[z] - P[z]).y)), 2)), 0.5));

				// X' = P' + u*(Q'-P') + v*Perpendicular(Q'-P') / || Q'-P'||
				X_prime = P_prime[z] + u * (Q_prime[z] - P_prime[z]) + v * perp(Q_prime[z] - P_prime[z]) /
					(pow((pow((((Q_prime[z] - P_prime[z]).x)), 2) + pow((((Q_prime[z] - P_prime[z]).y)), 2)), 0.5));

				// weight[z] = [ ( length[z]^p ) / ( a + dist[z] ) ] ^b
				// a = 1 , b = 2 , p = 0 
				if (u < 0)
					dist = pow(pow((X_prime - P_prime[z]).x, 2) + pow((X_prime - P_prime[z]).y, 2), 0.5);
				else if (u > 1)
					dist = pow(pow((X_prime - Q_prime[z]).x, 2) + pow((X_prime - Q_prime[z]).y, 2), 0.5);
				else
					dist = abs(v);
				weight = pow((pow(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2), 0.5) / (1 + dist)), 2);

				// XSum = Xsum + X'[z]*weight
				XSum21 = XSum21 + X_prime * weight;
				// WeightSum += weight[z]
				WeightSum21 = WeightSum21 + weight;

				// 1->2
				// u  = (X-P')*(Q'-P') / ||Q'-P'||^2
				u = (((Q_prime[z] - P_prime[z]).x*(X - P_prime[z]).x + (Q_prime[z] - P_prime[z]).y*(X - P_prime[z]).y)) /
					(pow((((Q_prime[z] - P_prime[z]).x)), 2) + pow((((Q_prime[z] - P_prime[z]).y)), 2));

				// v = (X-P')*Perpendicular(Q'-P') / ||Q'-P'||
				v = (((X - P_prime[z]).x*(Q_prime[z] - P_prime[z]).y - (X - P_prime[z]).y*(Q_prime[z] - P_prime[z]).x)) /
					(pow((pow((((Q_prime[z] - P_prime[z]).x)), 2) + pow((((Q_prime[z] - P_prime[z]).y)), 2)), 0.5));

				// X' = P + u*(Q-P) + v*Perpendicular(Q-P) / || Q-P||
				X_prime = P[z] + u * (Q[z] - P[z]) + v * perp(Q[z] - P[z]) /
					(pow((pow((((Q[z] - P[z]).x)), 2) + pow((((Q[z] - P[z]).y)), 2)), 0.5));

				// weight[z] = [ ( length[z]^p ) / ( a + dist[z] ) ] ^b
				// a = 1 , b = 2 , p = 0 
				if (u < 0)
					dist = pow(pow((X_prime - P_prime[z]).x, 2) + pow((X_prime - P_prime[z]).y, 2), 0.5);
				else if (u > 1)
					dist = pow(pow((X_prime - Q_prime[z]).x, 2) + pow((X_prime - Q_prime[z]).y, 2), 0.5);
				else
					dist = abs(v);
				weight = pow((pow(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2), 0.5) / (1 + dist)), 2);

				// XSum = Xsum + X'[z]*weight
				XSum12 = XSum12 + X_prime * weight;
				// WeightSum += weight[z]
				WeightSum12 = WeightSum12 + weight;
			}
			
			mapx_14.at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*0.75 + X*0.25).x;
			mapy_14.at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*0.75 + X*0.25).y;
			mapx_24.at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*0.5  + X*0.5 ).x;
			mapy_24.at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*0.5  + X*0.5 ).y;
			mapx_34.at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*0.25 + X*0.75).x;
			mapy_34.at<float>(Point(x, y)) = ((XSum21 / WeightSum21)*0.25 + X*0.75).y;

			mapx_14_2.at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*0.25 + X*0.75).x;
			mapy_14_2.at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*0.25 + X*0.75).y;
			mapx_24_2.at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*0.5 + X*0.5).x;
			mapy_24_2.at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*0.5 + X*0.5).y;
			mapx_34_2.at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*0.75 + X*0.25).x;
			mapy_34_2.at<float>(Point(x, y)) = ((XSum12 / WeightSum12)*0.75 + X*0.25).y;
		}
	}
	
	cv::remap(image2, two2one_14, mapx_14, mapy_14, INTER_LINEAR);
	cv::remap(image2, two2one_24, mapx_24, mapy_24, INTER_LINEAR);
	cv::remap(image2, two2one_34, mapx_34, mapy_34, INTER_LINEAR);

	cv::remap(image1, one2two_14, mapx_14_2, mapy_14_2, INTER_LINEAR);
	cv::remap(image1, one2two_24, mapx_24_2, mapy_24_2, INTER_LINEAR);
	cv::remap(image1, one2two_34, mapx_34_2, mapy_34_2, INTER_LINEAR);
	
	cv::addWeighted(one2two_14, 0.75, two2one_14, 0.25, 0.0, mix_14);
	cv::addWeighted(one2two_24, 0.5,  two2one_24, 0.5 , 0.0, mix_24);
	cv::addWeighted(one2two_34, 0.25, two2one_34, 0.75, 0.0, mix_34);

	cv::namedWindow("t = 0", WINDOW_AUTOSIZE);
	cv::namedWindow("t = 0.25", WINDOW_AUTOSIZE);
	cv::namedWindow("t = 0.5", WINDOW_AUTOSIZE);
	cv::namedWindow("t = 0.75", WINDOW_AUTOSIZE);
	cv::namedWindow("t = 1", WINDOW_AUTOSIZE);

	cv::imshow("t = 0", image1);
	cv::imshow("t = 0.25", mix_14);
	cv::imshow("t = 0.5", mix_24);
	cv::imshow("t = 0.75", mix_34);
	cv::imshow("t = 1", image2);

	cv::imwrite("t025.jpg", mix_14);
	cv::imwrite("t050.jpg", mix_24);
	cv::imwrite("t075.jpg", mix_34);

}

void CallBack(int event, int x, int y, int flags, void* userdata)
{
	// image1
	if (x >= img1_x && x <= (img1_x+image1.cols) && y >= img1_y && y <= (img1_y+image1.rows))
	{
		// press mouse
		if (event == EVENT_LBUTTONDOWN)
		{
			std::cout << "image1" << endl;
			std::cout << "start at (" << x - image1_original.x << ", " << y - image1_original.y << ")" << endl;
			image1_start = Point(x, y);
			img1_start_check = true;
		}
		// release mouse
		else if (event == EVENT_LBUTTONUP)
		{
			std::cout << "end at (" << x - image1_original.x << ", " << y - image1_original.y << ")" << endl;
			image1_end = Point(x, y);
			img1_end_check = true;
			if (img1_start_check == true && img1_end_check == true) {
				arrowedLine(OutputImage, image1_start, image1_end, Scalar(0, 255, 255), 3);
				P[image1_arrow_index] = image1_start - image1_original;
				Q[image1_arrow_index] = image1_end - image1_original;
				image1_arrow_index++;
				img1_start_check = false;
				img1_end_check = false;
			}
		}
	}
	// image2 
	else if (x >= img2_x && x <= (img2_x+image2.cols) && y >= img2_y && y <= (img2_y+image2.rows))
	{
		// press mouse 
		if (event == EVENT_LBUTTONDOWN)
		{
			std::cout << "image2" << endl;
			std::cout << "start at (" << x - image2_original.x << ", " << y - image2_original.y << ")" << endl;
			image2_start = Point(x, y);
			img2_start_check = true;
		}
		// release mouse
		else if (event == EVENT_LBUTTONUP)
		{
			std::cout << "end at (" << x - image2_original.x << ", " << y - image2_original.y << ")" << endl;
			image2_end = Point(x, y);
			img2_end_check = true;

			if (img2_start_check == true && img2_end_check == true) {
				arrowedLine(OutputImage, image2_start, image2_end, Scalar(0, 255, 255), 3);
				P_prime[image2_arrow_index] = image2_start - image2_original;
				Q_prime[image2_arrow_index] = image2_end - image2_original;
				image2_arrow_index++;
				img2_start_check = false;
				img2_end_check = false;
			}
		}
	}
	// morphing button 
	else if (x >= 325 && x < 485 && y >= 70 && y <= 150)
	{
		if (event == EVENT_LBUTTONDOWN)
			button_true = true;
	}

	if (button_true == true)
	{
		button_true = false;
		morphing();
	}
	cv::imshow("Morphing", OutputImage);
}

int main(int argc, char** argv)
{
	OutputImage.setTo(Scalar(100, 0, 0));
	image1 = imread("women.jpg");
	image2 = imread("cheetah.jpg");
	morph_button = imread("morph_button.png");

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

	cv::namedWindow("Morphing", WINDOW_AUTOSIZE);
	cv::setMouseCallback("Morphing", CallBack);
	cv::imshow("Morphing", OutputImage);
	cv::waitKey(0);
	cv::destroyWindow("Morphing");
	return 0;
}