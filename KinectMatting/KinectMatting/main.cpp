#include <iostream>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <Kinect.h>

using   namespace   std;
using   namespace   cv;

int main(void)
{
	IKinectSensor   * mySensor = nullptr;
	GetDefaultKinectSensor(&mySensor);
	mySensor->Open();

	//************************准备好彩色图像的Reader并获取尺寸*******************************

	int colorHeight = 0, colorWidth = 0;
	IColorFrameSource   * myColorSource = nullptr;
	IColorFrameReader   * myColorReader = nullptr;
	IFrameDescription   * myDescription = nullptr;
	{
		mySensor->get_ColorFrameSource(&myColorSource);

		myColorSource->OpenReader(&myColorReader);

		myColorSource->get_FrameDescription(&myDescription);
		myDescription->get_Height(&colorHeight);
		myDescription->get_Width(&colorWidth);

		myDescription->Release();
		myColorSource->Release();
	}

	//************************准备好深度图像的Reader并获取尺寸*******************************

	int depthHeight = 0, depthWidth = 0;
	IDepthFrameSource   * myDepthSource = nullptr;
	IDepthFrameReader   * myDepthReader = nullptr;
	{
		mySensor->get_DepthFrameSource(&myDepthSource);

		myDepthSource->OpenReader(&myDepthReader);

		myDepthSource->get_FrameDescription(&myDescription);
		myDescription->get_Height(&depthHeight);
		myDescription->get_Width(&depthWidth);

		myDescription->Release();
		myDepthSource->Release();
	}

	//************************准备好人体索引图像的Reader并获取尺寸****************************

	int bodyHeight = 0, bodyWidth = 0;
	IBodyIndexFrameSource   * myBodyIndexSource = nullptr;
	IBodyIndexFrameReader   * myBodyIndexReader = nullptr;
	{
		mySensor->get_BodyIndexFrameSource(&myBodyIndexSource);

		myBodyIndexSource->OpenReader(&myBodyIndexReader);

		myDepthSource->get_FrameDescription(&myDescription);
		myDescription->get_Height(&bodyHeight);
		myDescription->get_Width(&bodyWidth);

		myDescription->Release();
		myBodyIndexSource->Release();
	}

	//************************为各种图像准备buffer，并且开启Mapper*****************************

	UINT    colorDataSize = colorHeight * colorWidth;
	UINT    depthDataSize = depthHeight * depthWidth;
	UINT    bodyDataSize = bodyHeight * bodyWidth;
	Mat temp = imread("test.jpg"), background;               //获取背景图
	resize(temp, background, Size(colorWidth, colorHeight));   //调整至彩色图像的大小

	ICoordinateMapper   * myMaper = nullptr;                //开启mapper
	mySensor->get_CoordinateMapper(&myMaper);

	Mat colorData(colorHeight, colorWidth, CV_8UC4);        //准备buffer
	UINT16  * depthData = new UINT16[depthDataSize];
	BYTE    * bodyData = new BYTE[bodyDataSize];
	DepthSpacePoint * output = new DepthSpacePoint[colorDataSize];

	//************************把各种图像读进buffer里，然后进行处理*****************************

	while (1)
	{
		IColorFrame * myColorFrame = nullptr;
		while (myColorReader->AcquireLatestFrame(&myColorFrame) != S_OK);   //读取color图
		myColorFrame->CopyConvertedFrameDataToArray(colorDataSize * 4, colorData.data, ColorImageFormat_Bgra);
		myColorFrame->Release();

		IDepthFrame * myDepthframe = nullptr;
		while (myDepthReader->AcquireLatestFrame(&myDepthframe) != S_OK);   //读取depth图
		myDepthframe->CopyFrameDataToArray(depthDataSize, depthData);
		myDepthframe->Release();

		IBodyIndexFrame * myBodyIndexFrame = nullptr;                       //读取BodyIndex图
		while (myBodyIndexReader->AcquireLatestFrame(&myBodyIndexFrame) != S_OK);
		myBodyIndexFrame->CopyFrameDataToArray(bodyDataSize, bodyData);
		myBodyIndexFrame->Release();

		Mat copy = background.clone();                  //复制一份背景图来做处理
		if (myMaper->MapColorFrameToDepthSpace(depthDataSize, depthData, colorDataSize, output) == S_OK)
		{
			for (int i = 0; i < colorHeight; ++i)
			for (int j = 0; j < colorWidth; ++j)
			{
				DepthSpacePoint tPoint = output[i * colorWidth + j];    //取得彩色图像上的一点，此点包含了它对应到深度图上的坐标
				if (tPoint.X >= 0 && tPoint.X < depthWidth && tPoint.Y >= 0 && tPoint.Y < depthHeight)  //判断是否合法
				{
					int index = (int)tPoint.Y * depthWidth + (int)tPoint.X; //取得彩色图上那点对应在BodyIndex里的值(注意要强转)
					if (bodyData[index] <= 5)                   //如果判断出彩色图上某点是人体，就用它来替换背景图上对应的点
					{
						Vec4b   color = colorData.at<Vec4b>(i, j);
						copy.at<Vec3b>(i, j) = Vec3b(color[0], color[1], color[2]);
					}
				}
			}
			imshow("TEST", copy);
		}
		if (waitKey(30) == VK_ESCAPE)
			break;
	}
	delete[] depthData;         //记得各种释放
	delete[] bodyData;
	delete[] output;


	myMaper->Release();
	myColorReader->Release();
	myDepthReader->Release();
	myBodyIndexReader->Release();
	mySensor->Close();
	mySensor->Release();

	return  0;
}