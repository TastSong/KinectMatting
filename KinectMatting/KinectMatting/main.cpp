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

	//************************׼���ò�ɫͼ���Reader����ȡ�ߴ�*******************************

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

	//************************׼�������ͼ���Reader����ȡ�ߴ�*******************************

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

	//************************׼������������ͼ���Reader����ȡ�ߴ�****************************

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

	//************************Ϊ����ͼ��׼��buffer�����ҿ���Mapper*****************************

	UINT    colorDataSize = colorHeight * colorWidth;
	UINT    depthDataSize = depthHeight * depthWidth;
	UINT    bodyDataSize = bodyHeight * bodyWidth;
	Mat temp = imread("test.jpg"), background;               //��ȡ����ͼ
	resize(temp, background, Size(colorWidth, colorHeight));   //��������ɫͼ��Ĵ�С

	ICoordinateMapper   * myMaper = nullptr;                //����mapper
	mySensor->get_CoordinateMapper(&myMaper);

	Mat colorData(colorHeight, colorWidth, CV_8UC4);        //׼��buffer
	UINT16  * depthData = new UINT16[depthDataSize];
	BYTE    * bodyData = new BYTE[bodyDataSize];
	DepthSpacePoint * output = new DepthSpacePoint[colorDataSize];

	//************************�Ѹ���ͼ�����buffer�Ȼ����д���*****************************

	while (1)
	{
		IColorFrame * myColorFrame = nullptr;
		while (myColorReader->AcquireLatestFrame(&myColorFrame) != S_OK);   //��ȡcolorͼ
		myColorFrame->CopyConvertedFrameDataToArray(colorDataSize * 4, colorData.data, ColorImageFormat_Bgra);
		myColorFrame->Release();

		IDepthFrame * myDepthframe = nullptr;
		while (myDepthReader->AcquireLatestFrame(&myDepthframe) != S_OK);   //��ȡdepthͼ
		myDepthframe->CopyFrameDataToArray(depthDataSize, depthData);
		myDepthframe->Release();

		IBodyIndexFrame * myBodyIndexFrame = nullptr;                       //��ȡBodyIndexͼ
		while (myBodyIndexReader->AcquireLatestFrame(&myBodyIndexFrame) != S_OK);
		myBodyIndexFrame->CopyFrameDataToArray(bodyDataSize, bodyData);
		myBodyIndexFrame->Release();

		Mat copy = background.clone();                  //����һ�ݱ���ͼ��������
		if (myMaper->MapColorFrameToDepthSpace(depthDataSize, depthData, colorDataSize, output) == S_OK)
		{
			for (int i = 0; i < colorHeight; ++i)
			for (int j = 0; j < colorWidth; ++j)
			{
				DepthSpacePoint tPoint = output[i * colorWidth + j];    //ȡ�ò�ɫͼ���ϵ�һ�㣬�˵����������Ӧ�����ͼ�ϵ�����
				if (tPoint.X >= 0 && tPoint.X < depthWidth && tPoint.Y >= 0 && tPoint.Y < depthHeight)  //�ж��Ƿ�Ϸ�
				{
					int index = (int)tPoint.Y * depthWidth + (int)tPoint.X; //ȡ�ò�ɫͼ���ǵ��Ӧ��BodyIndex���ֵ(ע��Ҫǿת)
					if (bodyData[index] <= 5)                   //����жϳ���ɫͼ��ĳ�������壬���������滻����ͼ�϶�Ӧ�ĵ�
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
	delete[] depthData;         //�ǵø����ͷ�
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