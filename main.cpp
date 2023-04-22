#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define MASK_PATH = "";
#define OFFSET = 0;
#define M = 3;  // M deve ser ímpar, por ser a quantidade de linhas do filtro da mediana 
#define N = 11; // N deve ser ímpar, por ser a quantidade de colunas do filtro da mediana

using namespace cv;
using namespace std;

void rgbToYIQ(Mat&, Mat&);
void yiqToRgb(Mat&, Mat&);
void negativeY(Mat&, Mat&);
void negativeR(Mat&, Mat&);
void negativeG(Mat&, Mat&);
void negativeB(Mat&, Mat&);
void negativeRGB(Mat&, Mat&);
void medianFilterR(Mat&, Mat&, int, int);
void medianFilterG(Mat&, Mat&, int, int);
void medianFilterB(Mat&, Mat&, int, int);
void medianFilterRGB(Mat&, Mat&, int, int);
void padImage(Mat&, Mat&, int, int, int, int);
bool readMask(Mat&, vector<int>&, const char*);
bool correlation(Mat&, Mat&, const char*);

int main()
{
    Mat image = imread("testpat.1k.color2.tif"); // read the image file

    if (image.empty()) // check for failure
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    cout << "Image size: " << image.size() <<  endl;

    
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    int rows = image.rows;
    int cols = image.cols;

    Mat blue(rows, cols, CV_8UC1); // 2D array for blue channel
    Mat green(rows, cols, CV_8UC1); // 2D array for green channel
    Mat red(rows, cols, CV_8UC1); // 2D array for red channel
    Mat blueImage(rows, cols, CV_8UC3);
    Mat greenImage(rows, cols, CV_8UC3);
    Mat redImage(rows, cols, CV_8UC3);
    // iterate over each pixel and store its RGB value in the corresponding array
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            Vec3b intensity = image.at<Vec3b>(i, j);
            blue.at<uchar>(i, j) = intensity.val[0];
            green.at<uchar>(i, j) = intensity.val[1];
            red.at<uchar>(i, j) = intensity.val[2];
            blueImage.at<Vec3b>(i, j)[0] = intensity.val[0];
            blueImage.at<Vec3b>(i, j)[1] = 0;
            blueImage.at<Vec3b>(i, j)[2] = 0;
            greenImage.at<Vec3b>(i, j)[0] = 0;
            greenImage.at<Vec3b>(i, j)[1] = intensity.val[1];
            greenImage.at<Vec3b>(i, j)[2] = 0;
            redImage.at<Vec3b>(i, j)[0] = 0;
            redImage.at<Vec3b>(i, j)[1] = 0;
            redImage.at<Vec3b>(i, j)[2] = intensity.val[2];
        }
    }

    
    namedWindow("Blue Channel", WINDOW_AUTOSIZE);
    imshow("Blue Channel", blueImage);

    namedWindow("Green Channel", WINDOW_AUTOSIZE);
    imshow("Green Channel", greenImage);

    namedWindow("Red Channel", WINDOW_AUTOSIZE);
    imshow("Red Channel", redImage);

    Mat imageYIQ;
    Mat negAuxY;
    Mat negY;
    Mat negR;
    Mat negG;
    Mat negB;
    Mat negRGB;
    Mat restored;
    Mat medianR;
    Mat medianG;
    Mat medianB;
    Mat medianRGB;
    Mat paddedImage;
    Mat meanFilter;
    
    rgbToYIQ(image, imageYIQ);
    namedWindow("YIQ as BGR", WINDOW_AUTOSIZE);
    imshow("YIQ as BGR", imageYIQ);

    negativeY(imageYIQ, negAuxY);
    yiqToRgb(negAuxY, negY);
    namedWindow("Negative Y", WINDOW_AUTOSIZE);
    imshow("Negative Y", negY);

    negativeR(image, negR);
    namedWindow("Negative R", WINDOW_AUTOSIZE);
    imshow("Negative R", negR);

    negativeG(image, negG);
    namedWindow("Negative G", WINDOW_AUTOSIZE);
    imshow("Negative G", negG);

    negativeB(image, negB);
    namedWindow("Negative B", WINDOW_AUTOSIZE);
    imshow("Negative B", negB);

    negativeRGB(image, negRGB);
    namedWindow("Negative RGB", WINDOW_AUTOSIZE);
    imshow("Negative RGB", negRGB);


    medianFilterR(image, medianR, 5, 7);
    namedWindow("Median R", WINDOW_AUTOSIZE);
    imshow("Median R", medianR);

    medianFilterG(image, medianG, 5, 7);
    namedWindow("Median G", WINDOW_AUTOSIZE);
    imshow("Median G", medianG);

    medianFilterB(image, medianB, 5, 7);
    namedWindow("Median B", WINDOW_AUTOSIZE);
    imshow("Median B", medianB);

    medianFilterRGB(image, medianRGB, 5, 7);
    namedWindow("Median RGB", WINDOW_AUTOSIZE);
    imshow("Median RGB", medianRGB);

    padImage(image, paddedImage, 5, 11, 2, 5);
    namedWindow("Padded Image", WINDOW_AUTOSIZE);
    imshow("Padded Image", paddedImage);

    yiqToRgb(imageYIQ, restored);
    namedWindow("Restored RGB", WINDOW_AUTOSIZE);
    imshow("Restored RGB", restored);
    bool isCorrelationValid = correlation(image, meanFilter, "box.txt");
    if(isCorrelationValid){
        namedWindow("Box 3x3", WINDOW_AUTOSIZE);
        imshow("Box 3x3", meanFilter);
    }

    waitKey(0); // wait for a key press
    return 0;
}

void rgbToYIQ(Mat& src, Mat& dst) {
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)

    // Split the input image into 3 channels (R, G, B)
    /*vector<Mat> channels;
    split(src, channels);*/
    
    // Compute the YIQ color space values
    dst.create(src.size(), CV_32FC3);
    for (int k = 0; k < src.rows; k++) {
        for (int j = 0; j < src.cols; j++) {
            uchar r = src.at<Vec3b>(k, j)[2];
            uchar g = src.at<Vec3b>(k, j)[1];
            uchar b = src.at<Vec3b>(k, j)[0];

            float y = (0.299*r + 0.587*g + 0.114*b);
            float i = (0.596*r - 0.275*g - 0.321*b);
            float q = (0.212*r - 0.523*g + 0.311*b);

            dst.at<Vec3f>(k, j) = Vec3f(y, i, q);
        }
    }
}

void yiqToRgb(cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (YIQ)

    // Split the input image into 3 channels (Y, I, Q)
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    // Compute the RGB color space values
    dst.create(src.size(), CV_8UC3);
    for (int k = 0; k < src.rows; k++) {
        for (int j = 0; j < src.cols; j++) {
            float y = channels[0].at<float>(k, j);
            float i = channels[1].at<float>(k, j);
            float q = channels[2].at<float>(k, j);

            uchar r = y + 0.956*i + 0.621*q;
            uchar g = y - 0.272*i - 0.647*q;
            uchar b = y - 1.106*i + 1.703*q;

            // Normalize the RGB color values to 0-255 range
            r = (r > 255) ? 255 : r;
            g = (g > 255) ? 255 : g;
            b = (b > 255) ? 255 : b;
            
            r = (r < 0) ? 0 : r;
            g = (g < 0) ? 0 : g;
            b = (b < 0) ? 0 : b;

            dst.at<Vec3b>(k, j) = Vec3b(b,g,r);
        }
    }
}

void negativeY(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (YIQ)
    
    dst.create(src.size(), CV_32FC3);

    for(int k = 0; k < src.rows; k++){
        for(int j = 0; j < src.cols; j++){
            float y = 255.0 - src.at<Vec3f>(k,j)[0];
            float i = src.at<Vec3f>(k,j)[1];
            float q = src.at<Vec3f>(k,j)[1];

            dst.at<Vec3f>(k, j) = Vec3f(y, i, q);
        }
    }
}

void negativeR(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)

    dst.create(src.size(), CV_8UC3);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            uchar r = 255 - src.at<Vec3b>(i, j)[2];
            uchar g = src.at<Vec3b>(i, j)[1];
            uchar b = src.at<Vec3b>(i, j)[0];

            dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
}

void negativeG(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    
    dst.create(src.size(), CV_8UC3);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            uchar r = src.at<Vec3b>(i, j)[2];
            uchar g = 255 - src.at<Vec3b>(i, j)[1];
            uchar b = src.at<Vec3b>(i, j)[0];

            dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
}

void negativeB(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    
    dst.create(src.size(), CV_8UC3);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            uchar r = src.at<Vec3b>(i, j)[2];
            uchar g = src.at<Vec3b>(i, j)[1];
            uchar b = 255 - src.at<Vec3b>(i, j)[0];

            dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
}

void negativeRGB(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)

    dst.create(src.size(), CV_8UC3);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            uchar r = 255 - src.at<Vec3b>(i, j)[2];
            uchar g = 255 - src.at<Vec3b>(i, j)[1];
            uchar b = 255 - src.at<Vec3b>(i, j)[0];

            dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
}

uchar getMedian(vector<uchar>& window)
{
    // Sort the window
    sort(window.begin(), window.end());

    // return median value
    return (window.size() % 2) ? window[window.size() / 2] : ((window[window.size() / 2] + window[1 + window.size() / 2])/2);
}

void medianFilterR(Mat& src, Mat& dst, int m, int n){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    Mat paddedSrc;
    padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
    dst.create(src.size(), CV_8UC3);
    
    vector<uchar> window;

    for(int i = ((m-1)/2); i <  (src.rows - ((m-1)/2)); i++){
        for(int j = ((n-1)/2); j < (src.cols - ((n-1)/2)); j++){
            for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
                for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
                    window.push_back(paddedSrc.at<Vec3b>(k,l)[2]);//red channel
                }
            }
            dst.at<Vec3b>(i,j) = Vec3b(paddedSrc.at<Vec3b>(i,j)[0], paddedSrc.at<Vec3b>(i,j)[1], getMedian(window));
            window.clear();
        }
    }
}

void medianFilterG(Mat& src, Mat& dst, int m, int n){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    Mat paddedSrc;
    padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
    dst.create(src.size(), CV_8UC3);
    
    vector<uchar> window;

    for(int i = ((m-1)/2); i <  (src.rows - ((m-1)/2)); i++){
        for(int j = ((n-1)/2); j < (src.cols - ((n-1)/2)); j++){
            for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
                for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
                    window.push_back(paddedSrc.at<Vec3b>(k,l)[1]);//green channel
                }
            }
            dst.at<Vec3b>(i,j) = Vec3b(paddedSrc.at<Vec3b>(i,j)[0], getMedian(window), paddedSrc.at<Vec3b>(i,j)[2]);
            window.clear();
        }
    }
}

void medianFilterB(Mat& src, Mat& dst, int m, int n){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    Mat paddedSrc;
    padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
    dst.create(src.size(), CV_8UC3);
    
    vector<uchar> window;

    for(int i = ((m-1)/2); i <  (src.rows - ((m-1)/2)); i++){
        for(int j = ((n-1)/2); j < (src.cols - ((n-1)/2)); j++){
            for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
                for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
                    window.push_back(paddedSrc.at<Vec3b>(k,l)[0]);//blue channel
                }
            }
            dst.at<Vec3b>(i,j) = Vec3b(getMedian(window), paddedSrc.at<Vec3b>(i,j)[1], paddedSrc.at<Vec3b>(i,j)[2]);
            window.clear();
        }
    }
}

void medianFilterRGB(Mat& src, Mat& dst, int m, int n){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    Mat paddedSrc;
    padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
    dst.create(src.size(), CV_8UC3);
    
    vector<uchar> windowR, windowG, windowB;

    for(int i = ((m-1)/2); i <  (src.rows - ((m-1)/2)); i++){
        for(int j = ((n-1)/2); j < (src.cols - ((n-1)/2)); j++){
            for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
                for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
                    windowB.push_back(paddedSrc.at<Vec3b>(k,l)[0]);//blue channel
                    windowG.push_back(paddedSrc.at<Vec3b>(k,l)[1]);//green channel
                    windowR.push_back(paddedSrc.at<Vec3b>(k,l)[2]);//red channel
                }
            }
            dst.at<Vec3b>(i,j) = Vec3b(getMedian(windowB), getMedian(windowG), getMedian(windowR));
            windowB.clear();
            windowG.clear();
            windowR.clear();
        }
    }
}

// void medianFilterR(Mat& src, Mat& dst, int m, int n){
//     CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
//     // Mat paddedSrc;
//     // padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
//     dst.create(src.size(), CV_8UC3);
    
//     vector<uchar> window;

//     for(int i = ((m-1)/2); i <  (src.rows - m); i++){
//         for(int j = ((n-1)/2); j < (src.cols - n); j++){
//             for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
//                 for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
//                     window.push_back(src.at<Vec3b>(k,l)[2]);//red channel
//                 }
//             }
//             dst.at<Vec3b>(i,j) = Vec3b(src.at<Vec3b>(i,j)[0], src.at<Vec3b>(i,j)[1], getMedian(window));
//             window.clear();
//         }
//     }
// }

// void medianFilterG(Mat& src, Mat& dst, int m, int n){
//     CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
//     // Mat paddedSrc;
//     // padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
//     dst.create(src.size(), CV_8UC3);
    
//     vector<uchar> window;

//     for(int i = ((m-1)/2); i <  (src.rows - m); i++){
//         for(int j = ((n-1)/2); j < (src.cols - n); j++){
//             for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
//                 for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
//                     window.push_back(src.at<Vec3b>(k,l)[1]);//green channel
//                 }
//             }
//             dst.at<Vec3b>(i,j) = Vec3b(src.at<Vec3b>(i,j)[0], getMedian(window), src.at<Vec3b>(i,j)[2]);
//             window.clear();
//         }
//     }
// }

// void medianFilterB(Mat& src, Mat& dst, int m, int n){
//     CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
//     // Mat paddedSrc;
//     // padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
//     dst.create(src.size(), CV_8UC3);
    
//     vector<uchar> window;

//     for(int i = ((m-1)/2); i <  (src.rows - m); i++){
//         for(int j = ((n-1)/2); j < (src.cols - n); j++){
//             for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
//                 for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
//                     window.push_back(src.at<Vec3b>(k,l)[0]);//blue channel
//                 }
//             }
//             dst.at<Vec3b>(i,j) = Vec3b(getMedian(window), src.at<Vec3b>(i,j)[1], src.at<Vec3b>(i,j)[2]);
//             window.clear();
//         }
//     }
// }

// void medianFilterRGB(Mat& src, Mat& dst, int m, int n){
//     CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
//     //Mat paddedSrc;
//     //padImage(src, paddedSrc, m, n, ((m-1)/2), ((n-1)/2));
//     dst.create(src.size(), CV_8UC3);
    
//     vector<uchar> windowR, windowG, windowB;

//     for(int i = ((m-1)/2); i <  (src.rows - m); i++){
//         for(int j = ((n-1)/2); j < (src.cols - n); j++){
//             for(int k = i - ((m-1)/2); k <= i + ((m-1)/2); k++){
//                 for(int l = j - ((n-1)/2); l <= j + ((n-1)/2); l++){
//                     windowB.push_back(src.at<Vec3b>(k,l)[0]);//blue channel
//                     windowG.push_back(src.at<Vec3b>(k,l)[1]);//green channel
//                     windowR.push_back(src.at<Vec3b>(k,l)[2]);//red channel
//                 }
//             }
//             dst.at<Vec3b>(i,j) = Vec3b(getMedian(windowB), getMedian(windowG), getMedian(windowR));
//             windowB.clear();
//             windowG.clear();
//             windowR.clear();
//         }
//     }
// }

void padImage(Mat& src, Mat& dst, int m, int n, int pi, int pj){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
    dst.create(src.rows + m - 1, src.cols + n - 1, CV_8UC3);

    for(int i = 0; i < pi; i++){
        for(int j = 0; j < dst.cols; j++){
            dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    for(int i = src.rows + pi; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < pj; j++){
            dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    for(int i = 0; i < dst.rows; i++){
        for(int j = src.cols + pj; j < dst.cols; j++){
            dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    for(int i = pi; i < (dst.rows - m + pi); i++){
        for(int j = pj; j < (dst.cols - n  + pj); j++){

            uchar r = src.at<Vec3b>(i, j)[2];
            uchar g = src.at<Vec3b>(i, j)[1];
            uchar b = src.at<Vec3b>(i, j)[0];

            dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
}

double dotProduct(Mat& mask, Mat& window, bool& flag){
    double result = 0;
    flag = (mask.rows == window.rows && mask.cols == window.cols);
    if(flag){
        for(int i = 0; i < mask.rows; i++){
            for(int j = 0; j < mask.cols; j++){
                result += mask.at<Vec<double,1>>(i,j)[0] * window.at<Vec<double,1>>(i,j)[0];
            }
        }
    }
    return result;
}

bool readMask(Mat& Mask, vector<int>& args, const char *path){
    fstream file;
    vector<string> lines;

    file.open(path, ios::in);    // reads data and stores into a string vector
    try{
        if(!file.is_open()){
            throw "File could not be open!\n";
        }
        
        for(string line; getline(file, line); ){
            lines.push_back(line);
        }

    }catch(const char *error){
        cout << error;
    }
    
    file.close();

    stringstream sstream(lines[0]);
    string  sstreamValue;

    while(getline(sstream, sstreamValue, ' ')){
        args.push_back(stoi(sstreamValue));
    }

    if(args.size() != 5){
        args.clear();
        cout << "Invalid Mask" << endl;
        return false;
    }

    Mask.create(args.at(0), args.at(1), CV_32FC1);

    for(int i = 1; i <= args.at(0); i++){
        stringstream mySstream(lines[i]);
        int j = 0;
        while(getline(mySstream, sstreamValue, ' ')){
            //cout << "i: " << i << ", j: " << j << endl;
            Mask.at<Vec<double, 1>>(i - 1,j) =  Vec<double,1>(stod(sstreamValue));
            //cout << "[" << Mask.at<Vec<double,1>>(i,j)[0] << "]";
            j++;
        }
        if(j != args.at(1)){
            Mask.~Mat();
            args.clear();
            cout << "Invalid Mask" << endl;
            return false;
        }
    }

    // for(int i = 0; i < Mask.rows; i++){
    //     for(int j = 0; j < Mask.cols; j++){
    //         cout << "[" << Mask.at<Vec<double,1>>(i,j)[0] << "]";
    //     }
    //     cout << "\n";
    // }

    return true;
}

bool correlation(Mat& src, Mat& dst, const char* path){
    Mat mask;
    vector<int> args;
    
    if(readMask(mask, args, path)){
        if(args.size() == 5){// 5 args are needed for the correlation, being the number of rows(m), number of columns(n), pivot row(pi), pivot cloumn(pj) and offset value.
            if(mask.rows == args.at(0) && mask.cols == args.at(1)){
                CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
                Mat paddedSrc;
                padImage(src, paddedSrc, args.at(0), args.at(1), args.at(2), args.at(3));
                dst.create(src.size(), CV_8UC3);

                Mat windowB, windowG, windowR;
                windowB.create(args.at(0), args.at(1), CV_32FC1);
                windowG.create(args.at(0), args.at(1), CV_32FC1);
                windowR.create(args.at(0), args.at(1), CV_32FC1);
                int r, g, b;
                bool flagR, flagG, flagB;

                int i = 0;
                int j = 0;

                while(i < src.rows){
                    while(j < src.cols){
                        for(int k = 0; k < mask.rows; k++){
                            for(int l = 0; l < mask.cols; l++){
                                windowB.at<Vec<double, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k + args.at(0) - args.at(2), j + l + args.at(1) - args.at(3))[0];//blue channel
                                windowG.at<Vec<double, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k + args.at(0) - args.at(2), j + l + args.at(1) - args.at(3))[1];//green channel
                                windowR.at<Vec<double, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k + args.at(0) - args.at(2), j + l + args.at(1) - args.at(3))[2];//red channel
                            }
                        }

                        // for(int k = 0; k < windowB.rows; k++){
                        //     for(int l = 0; l < windowB.cols; l++){
                        //         cout << "[" << windowB.at<Vec<double,1>>(k,l)[0] << "]";
                        //     }
                        //     cout << "\n";
                        // }

                        b = dotProduct(mask, windowB, flagB) + args.at(4);
                        g = dotProduct(mask, windowG, flagG) + args.at(4);
                        r = dotProduct(mask, windowR, flagR) + args.at(4);

                        cout << "b: " << b << ", g: " << g << ", r: " << r << endl;

                        b = (b < 0) ? 0 : b;
                        g = (g < 0) ? 0 : g;
                        r = (r < 0) ? 0 : r;
                        b = (b > 255) ? 255 : b;
                        g = (g > 255) ? 255 : g;
                        r = (r > 255) ? 255 : r;
                        dst.at<Vec3b>(i,j) = Vec3b(b, g, r);
                        if(!flagR || !flagG || !flagB)
                            return false;
                        j++;
                    }
                    i++;
                }

                return true;

                // for(int i = args.at(2); i < (paddedSrc.rows - args.at(2) + args.at(0)); i++){
                //     for(int j = args.at(3); j < (paddedSrc.cols - args.at(3) + args.at(1)); j++){
                //         for(int k = i - args.at(2); k < i + args.at(0) - args.at(2) + 1; k++){
                //             for(int l = j - args.at(3); l < j + args.at(1) - args.at(3) + 1; l++){
                //                 windowB.at<Vec<double, 1>>(k - i + args.at(2), l - j + args.at(3)) = paddedSrc.at<Vec3b>(k,l)[0];//blue channel
                //                 windowG.at<Vec<double, 1>>(k - i + args.at(2), l - j + args.at(3)) = paddedSrc.at<Vec3b>(k,l)[1];//green channel
                //                 windowR.at<Vec<double, 1>>(k - i + args.at(2), l - j + args.at(3)) = paddedSrc.at<Vec3b>(k,l)[2];//red channel
                //             }
                //         }

                //         b = windowB.dot(mask);
                //         g = windowG.dot(mask);
                //         r = windowR.dot(mask);

                //         cout << "b: " << b << ", g: " << g << ", r: " << r << endl;

                //         b = (b < 0) ? 0 : b;
                //         g = (g < 0) ? 0 : g;
                //         r = (r < 0) ? 0 : r;
                //         b = (b > 255) ? 255 : b;
                //         g = (g > 255) ? 255 : g;
                //         r = (r > 255) ? 255 : r;
                //         dst.at<Vec3b>(i,j) = Vec3b(b, g, r);
                //         if(!flagB || !flagG || !flagR)
                //             return false;
                //     }
                // }
                // return true;
            }
        }
    }
    return false;
}