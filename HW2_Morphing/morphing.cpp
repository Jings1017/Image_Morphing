#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace cv;
using namespace std;

Mat OutputImage = Mat::zeros(Size(800, 300), CV_8UC3);
Mat image1, image2, morph_button;

Point des1, des2, src1, src2;
Point des_initial(50, 70), source_initial(500, 70);

bool des1_true, des2_true, source1_true, source2_true;
bool button_true = false;

int des_arrow = 0, source_arrow = 0;

Point P[50], Q[50], P_prime[50], Q_prime[50];

Point perpendicular(Point p) {
	Point transform(p.y, -p.x);
	return transform;
}

void StartToMorph() {

	cv::namedWindow("t = 0", 1);
	cv::namedWindow("t = 0.25", 1);
	cv::namedWindow("t = 0.5", 1);
	cv::namedWindow("t = 0.75", 1);
	cv::namedWindow("t = 1", 1);

	Mat one2two_14, one2two_24, one2two_34;
	Mat two2one_14, two2one_24, two2one_34;
	Mat mix_14, mix_24, mix_34;

	Mat mapx_14, mapy_14;
	Mat mapx_24, mapy_24;
	Mat mapx_34, mapy_34;

	Mat mapx_14_2, mapy_14_2;
	Mat mapx_24_2, mapy_24_2;
	Mat mapx_34_2, mapy_34_2;

	Point X, X_prime, XSum;
	double u, v, weight, WeightSum, dist;
	int arrow_num;

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

	for (int x = 0; x < 255; x++) {
		// 2->1
		for (int y = 0; y < 189; y++) {
			X.x = x;
			X.y = y;
			XSum.x = 0;
			XSum.y = 0;
			u = 0;
			v = 0;
			weight = 0;
			WeightSum = 0.0;

			arrow_num = (des_arrow > source_arrow) ? source_arrow : des_arrow;

			for (int z = 0; z < arrow_num; z++) {

				// u  = (X-P)*(Q-P) / ||Q-P||^2
				u = ((double)((Q[z] - P[z]).x * (X - P[z]).x + (Q[z] - P[z]).y * (X - P[z]).y)) /
					(pow((double)(((Q[z] - P[z]).x)), 2) + pow((double)(((Q[z] - P[z]).y)), 2));

				// v = (X-P)*Perpendicular(Q-P) / ||Q-P||
				v = ((double)((X - P[z]).x * (Q[z] - P[z]).y - (X - P[z]).y * (Q[z] - P[z]).x)) /
					(pow((pow((double)(((Q[z] - P[z]).x)), 2) + pow((double)(((Q[z] - P[z]).y)), 2)), 0.5));

				// X' = P' + u*(Q'-P') + v*Perpendicular(Q'-P') / || Q'-P'||
				X_prime = P_prime[z] + u * (Q_prime[z] - P_prime[z]) + v * perpendicular(Q_prime[z] - P_prime[z]) /
					(pow((pow((double)(((Q_prime[z] - P_prime[z]).x)), 2) + pow((double)(((Q_prime[z] - P_prime[z]).y)), 2)), 0.5));

				// weight[i] = [ ( length[i]^p ) / ( a + dist[i] ) ] ^b
				// a = 1 , b = 2 , p = 0 
				if (u < 0)
					dist = pow(pow((X_prime - P_prime[z]).x, 2) + pow((X_prime - P_prime[z]).y, 2), 0.5);
				else if (u > 1)
					dist = pow(pow((X_prime - Q_prime[z]).x, 2) + pow((X_prime - Q_prime[z]).y, 2), 0.5);
				else
					dist = abs(v);
				weight = pow((pow(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2), 0.5) / (1 + dist)), 2);

				// XSum = Xsum + X'[i]*weight
				XSum = XSum + X_prime * weight;
				// WeightSum += weight[i]
				WeightSum = WeightSum + weight;
			}
			mapx_14.at<float>(y, x) = (((XSum / WeightSum) - X)*0.75 + X).x;
			mapy_14.at<float>(y, x) = (((XSum / WeightSum) - X)*0.75 + X).y;
			mapx_24.at<float>(y, x) = (((XSum / WeightSum) - X)*0.5 + X).x;
			mapy_24.at<float>(y, x) = (((XSum / WeightSum) - X)*0.5 + X).y;
			mapx_34.at<float>(y, x) = (((XSum / WeightSum) - X)*0.25 + X).x;
			mapy_34.at<float>(y, x) = (((XSum / WeightSum) - X)*0.25 + X).y;
		}

		// 1->2
		for (int y = 0; y < 189; y++) {
			X.x = x;
			X.y = y;
			XSum.x = 0;
			XSum.y = 0;
			u = 0;
			v = 0;
			weight = 0;
			WeightSum = 0.0;

			for (int z = 0; z < arrow_num; z++) {
				// u  = (X-P)*(Q-P) / ||Q-P||^2
				u = ((double)((Q_prime[z] - P_prime[z]).x*(X - P_prime[z]).x +
					(Q_prime[z] - P_prime[z]).y*(X - P_prime[z]).y)) /
					(pow((double)(((Q_prime[z] - P_prime[z]).x)), 2) + pow((double)(((Q_prime[z] - P_prime[z]).y)), 2));

				// v = (X-P)*Perpendicular(Q-P) / ||Q-P||
				v = ((double)((X - P_prime[z]).x*(Q_prime[z] - P_prime[z]).y -
					(X - P_prime[z]).y*(Q_prime[z] - P_prime[z]).x)) /
					(pow((pow((double)(((Q_prime[z] - P_prime[z]).x)), 2) + pow((double)(((Q_prime[z] - P_prime[z]).y)), 2)), 0.5));

				// X' = P' + u*(Q'-P') + v*Perpendicular(Q'-P') / || Q'-P'||
				X_prime = P[z] + u * (Q[z] - P[z]) + v * perpendicular(Q[z] - P[z]) /
					(pow((pow((double)(((Q[z] - P[z]).x)), 2) + pow((double)(((Q[z] - P[z]).y)), 2)), 0.5));

				// weight[i] = [ ( length[i]^p ) / ( a + dist[i] ) ] ^b
				// a = 1 , b = 2 , p = 0 
				if (u < 0)
					dist = pow(pow((X_prime - P_prime[z]).x, 2) + pow((X_prime - P_prime[z]).y, 2), 0.5);
				else if (u > 1)
					dist = pow(pow((X_prime - Q_prime[z]).x, 2) + pow((X_prime - Q_prime[z]).y, 2), 0.5);
				else
					dist = abs(v);
				weight = pow((pow(pow((Q[z] - P[z]).x, 2) + pow((Q[z] - P[z]).y, 2), 0.5) / (1 + dist)), 2);

				// XSum = Xsum + X'[i]*weight
				XSum = XSum + X_prime * weight;
				// WeightSum += weight[i]
				WeightSum = WeightSum + weight;
			}
			mapx_14_2.at<float>(y, x) = (((XSum / WeightSum) - X)*0.25 + X).x;
			mapy_14_2.at<float>(y, x) = (((XSum / WeightSum) - X)*0.25 + X).y;
			mapx_24_2.at<float>(y, x) = (((XSum / WeightSum) - X)*0.5 + X).x;
			mapy_24_2.at<float>(y, x) = (((XSum / WeightSum) - X)*0.5 + X).y;
			mapx_34_2.at<float>(y, x) = (((XSum / WeightSum) - X)*0.75 + X).x;
			mapy_34_2.at<float>(y, x) = (((XSum / WeightSum) - X)*0.75 + X).y;
		}
	}

	remap(image2, two2one_14, mapx_14, mapy_14, INTER_LINEAR);
	remap(image2, two2one_24, mapx_24, mapy_24, INTER_LINEAR);
	remap(image2, two2one_34, mapx_34, mapy_34, INTER_LINEAR);

	remap(image1, one2two_14, mapx_14_2, mapy_14_2, INTER_LINEAR);
	remap(image1, one2two_24, mapx_24_2, mapy_24_2, INTER_LINEAR);
	remap(image1, one2two_34, mapx_34_2, mapy_34_2, INTER_LINEAR);

	addWeighted(one2two_14, 0.75, two2one_14, 0.25, 0.0, mix_14);
	addWeighted(one2two_24, 0.5, two2one_24, 0.5, 0.0, mix_24);
	addWeighted(one2two_34, 0.25, two2one_34, 0.75, 0.0, mix_34);

	cv::imshow("t = 0", image1);
	cv::imshow("t = 0.25", mix_14);
	cv::imshow("t = 0.5", mix_24);
	cv::imshow("t = 0.75", mix_34);
	cv::imshow("t = 1", image2);
}

void CallBack(int event, int x, int y, int flags, void* userdata)
{
	// image1
	if (x >= 50 && x <= 305 && y >= 70 && y <= 259)
	{
		// press mouse
		if (event == EVENT_LBUTTONDOWN)
		{
			std::cout << "pressed at (" << x - des_initial.x << ", " << y - des_initial.y << ")" << endl;
			des1 = Point(x, y);
			des1_true = true;
		}
		// release mouse
		else if (event == EVENT_LBUTTONUP)
		{
			std::cout << "released at (" << x - des_initial.x << ", " << y - des_initial.y << ")" << endl;
			des2 = Point(x, y);
			des2_true = true;
			if (des1_true == true && des2_true == true) {
				// draw the arrowed line 
				arrowedLine(OutputImage, des1, des2, Scalar(255, 255, 255), 2);
				P[des_arrow] = des1 - des_initial;
				Q[des_arrow] = des2 - des_initial;
				des_arrow++;
				des1_true = false;
				des2_true = false;
			}
		}
	}
	// image2 
	else if (x >= 500 && x <= 755 && y >= 70 && y <= 259)
	{
		// press mouse 
		if (event == EVENT_LBUTTONDOWN)
		{
			std::cout << "pressed at (" << x - source_initial.x << ", " << y - source_initial.y << ")" << endl;
			src1 = Point(x, y);
			source1_true = true;
		}
		// release mouse
		else if (event == EVENT_LBUTTONUP)
		{
			std::cout << "released at (" << x - source_initial.x << ", " << y - source_initial.y << ")" << endl;
			src2 = Point(x, y);
			source2_true = true;
			if (source1_true == true && source2_true == true) {
				arrowedLine(OutputImage, src1, src2, Scalar(255, 255, 255), 2);
				P_prime[source_arrow] = src1 - source_initial;
				Q_prime[source_arrow] = src2 - source_initial;
				source_arrow++;
				source1_true = false;
				source2_true = false;
			}
		}
	}
	// morphing button 
	else if (x >= 325 && x < 485 && y >= 70 && y <= 150)
	{
		if (event == EVENT_LBUTTONDOWN)
			button_true = true;
	}
	// just show the path
	else if (event == EVENT_RBUTTONDOWN)
	{
		std::cout << "destination:" << endl;
		for (int a = 0; a < des_arrow; a++)
			std::cout << "(" << P[a].x << "," << P[a].y << ") to (" << Q[a].x << "," << Q[a].y << ")" << endl;

		std::cout << "source:" << endl;
		for (int a = 0; a < source_arrow; a++)
			std::cout << "(" << P_prime[a].x << "," << P_prime[a].y << ") to (" << Q_prime[a].x << "," << Q_prime[a].y << ")" << endl;
	}
	// press the morphing button
	if (button_true == true)
	{
		button_true = false;
		StartToMorph();
	}
	cv::imshow("Morphing", OutputImage);
}

int main(int argc, char** argv)
{
	image1 = imread("women.jpg");
	image2 = imread("cheetah.jpg");
	morph_button = imread("morph_button.png");
	resize(morph_button, morph_button, Size(160, 80));

	if (!image1.data || !image2.data || !morph_button.data)
	{
		printf("image load error \n");
		exit(-1);
	}

	Rect R1(50, 70, image1.cols, image1.rows);
	Rect R2(500, 70, image2.cols, image2.rows);
	Rect button_R(325, 70, 160, 80);
	image1.copyTo(OutputImage(R1));
	image2.copyTo(OutputImage(R2));
	morph_button.copyTo(OutputImage(button_R));

	cv::putText(OutputImage, "left image", Point(120, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 2);
	cv::putText(OutputImage, "right image", Point(550, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 2);

	cv::namedWindow("Morphing", 1);
	cv::setMouseCallback("Morphing", CallBack, NULL);
	cv::imshow("Morphing", OutputImage);
	cv::waitKey(0);
	cv::destroyWindow("Morphing");
	return 0;
}