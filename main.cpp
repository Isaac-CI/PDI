#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

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
void histogramExpansion(Mat&, Mat&);
void showImage(Mat&, const char*);
bool readMask(vector<float>&, vector<int>&, const char*);
bool correlation_abs(Mat&, Mat&, const char*);
bool correlation_sat(Mat&, Mat&, const char*);

int main()
{
    Mat image = imread("inputs\\testpat.1k.color2.tif"); // read the image file

    if (image.empty()) // check for failure
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat imageYIQ;
    cout << "RGB to YIQ" << endl;
    rgbToYIQ(image, imageYIQ);
    imwrite("outputs\\YIQ_as_BGR.png", imageYIQ);

    Mat negAuxY;
    Mat negY;
    cout << "Negative Y" << endl;
    negativeY(imageYIQ, negAuxY);
    yiqToRgb(negAuxY, negY);
    imwrite("outputs\\negative_Y.png", negY);

    Mat negR;
    cout << "Negative R" << endl;
    negativeR(image, negR);
    imwrite("outputs\\negative_R.png", negR);

    Mat negG;
    cout << "Negative G" << endl;
    negativeG(image, negG);
    imwrite("outputs\\negative_G.png", negG);

    Mat negB;
    cout << "Negative B" << endl;
    negativeB(image, negB);
    imwrite("outputs\\negative_B.png", negB);

    Mat negRGB;
    cout << "Negative RGB" << endl;
    negativeRGB(image, negRGB);
    imwrite("outputs\\negative_RGB.png", negRGB);

    Mat medianR;
    cout << "Median R" << endl;
    medianFilterR(image, medianR, 5, 7);
    imwrite("outputs\\median_R.png", medianR);

    Mat medianG;
    cout << "Median G" << endl;
    medianFilterG(image, medianG, 11, 7);
    imwrite("outputs\\median_G.png", medianG);

    Mat medianB;
    cout << "Median B" << endl;
    medianFilterB(image, medianB, 25, 1);
    imwrite("outputs\\median_B.png", medianB);

    Mat medianRGB;
    cout << "Median RGB" << endl;
    medianFilterRGB(image, medianRGB, 1, 25);
    imwrite("outputs\\median_RGB.png", medianRGB);
    
    Mat paddedImage;
    cout << "Padding Image" << endl;
    padImage(image, paddedImage, 25, 25, 12, 12);
    imwrite("outputs\\padded_image.png", paddedImage);

    Mat restored;
    cout << "Restoring Original Image" << endl;
    yiqToRgb(imageYIQ, restored);
    imwrite("outputs\\restored_image.png", restored);

    Mat box1x11;
    Mat box11x1;
    cout << "Mean11x1(Mean 1x11)" << endl;
    auto cascatedBoxStart = chrono::high_resolution_clock::now();
    bool isCorrelationValid = correlation_sat(image, box1x11, "masks\\box1x11.txt");
    if(isCorrelationValid){
        isCorrelationValid = correlation_sat(box1x11, box11x1, "masks\\box11x1.txt");
        if(isCorrelationValid){
            auto cascatedBoxFinish = chrono::high_resolution_clock::now();
            auto cascatedBoxDuration = chrono::duration_cast<chrono::microseconds>(cascatedBoxFinish - cascatedBoxStart);
            imwrite("outputs\\box_11x1(box 1x11).png", box11x1);
            cout << "Tempo de processamento dos filtros box 1x11 e 11x1 em cascata, em us: " << cascatedBoxDuration.count() << "us" << endl;
        }else
            cout << "invalid correlation 11x1" << endl;
    }else
        cout << "invalid correlation 1x11" << endl;

    Mat box11x11;
    cout << "Mean 11x11" << endl;
    auto boxStart = chrono::high_resolution_clock::now();
    isCorrelationValid = correlation_sat(image, box11x11, "masks\\box11x11.txt");
    if(isCorrelationValid){
        auto boxFinish = chrono::high_resolution_clock::now();
        auto boxDuration = chrono::duration_cast<chrono::microseconds>(boxFinish - boxStart);
        imwrite("outputs\\box_11x11.png", box11x11);
        cout << "Tempo de processamento do filtro box 11x11, em us: " << boxDuration.count() << "us" << endl;
    }else
        cout << "invalid correlation 11x11" << endl;

    Mat sum2x4;
    cout << "Sum 2x4" << endl;
    isCorrelationValid = correlation_sat(image, sum2x4, "masks\\sum2x4.txt");
    if(isCorrelationValid){
        imwrite("outputs\\sum_2x4.png", sum2x4);
    }else
        cout << "invalid correlation" << endl;

    Mat sum4x2;
    cout << "Sum 4x2" << endl;
    isCorrelationValid = correlation_sat(image, sum4x2, "masks\\sum4x2.txt");
    if(isCorrelationValid){
        imwrite("outputs\\sum_4x2.png", sum4x2);
    }else
        cout << "invalid correlation" << endl;

    Mat emboss_0;
    cout << "Emboss offset: 0" << endl;
    isCorrelationValid = correlation_abs(image, emboss_0, "masks\\emboss.txt");
    if(isCorrelationValid){
        imwrite("outputs\\emboss.png", emboss_0);
    }

    Mat emboss_64;
    cout << "Emboss offset: 64" << endl;
    isCorrelationValid = correlation_abs(image, emboss_64, "masks\\emboss_64.txt");
    if(isCorrelationValid){
        imwrite("outputs\\emboss_64.png", emboss_64);
    }    

    Mat emboss_128;
    cout << "Emboss offset: 128" << endl;
    isCorrelationValid = correlation_abs(image, emboss_128, "masks\\emboss_128.txt");
    if(isCorrelationValid){
        imwrite("outputs\\emboss_128.png", emboss_128);
    }

    Mat vSobel, vSobelHistogramExpansion;
    cout << "Vertical Sobel" << endl;
    isCorrelationValid = correlation_abs(image, vSobel, "masks\\vSobel.txt");
        if(isCorrelationValid){
        imwrite("outputs\\vertical_sobel.png", vSobel);
        histogramExpansion(vSobel, vSobelHistogramExpansion);
        imwrite("outputs\\vertical_sobel_histogram_expansion.png", vSobelHistogramExpansion);
    }

    Mat hSobel, hSobelHistogramExpansion;
    cout << "Horizontal Sobel" << endl;
    isCorrelationValid = correlation_abs(image, hSobel, "masks\\hSobel.txt");
        if(isCorrelationValid){
        imwrite("outputs\\horizontal_sobel.png", hSobel);
        histogramExpansion(hSobel, hSobelHistogramExpansion);
        imwrite("outputs\\horizontal_sobel_histogram_expansion.png", hSobelHistogramExpansion);
    }
    showImage(image, "Original");
    showImage(paddedImage, "Padded Image");
    showImage(restored, "Restored");
    waitKey(0);
    return 0;
}

void rgbToYIQ(Mat& src, Mat& dst) {
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)

    // Split the input image into 3 channels (R, G, B)
    vector<Mat> channels;
    split(src, channels);
    
    // Compute the YIQ color space values
    dst.create(src.size(), CV_32FC3);
    for (int k = 0; k < src.rows; k++) {
        for (int j = 0; j < src.cols; j++) {
            uchar r = channels[2].at<uchar>(k, j);
            uchar g = channels[1].at<uchar>(k, j);
            uchar b = channels[0].at<uchar>(k, j);

            float y = (0.299*r + 0.587*g + 0.114*b);
            float i = (0.596*r - 0.274*g - 0.322*b);
            float q = (0.211*r - 0.523*g + 0.312*b);

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

            float r = y + 0.956*i + 0.621*q;
            float g = y - 0.272*i - 0.647*q;
            float b = y - 1.106*i + 1.703*q;

            // Normalize the RGB color values to 0-255 range
            r = (r > 255) ? 255 : r;
            g = (g > 255) ? 255 : g;
            b = (b > 255) ? 255 : b;
            
            r = (r < 0) ? 0 : r;
            g = (g < 0) ? 0 : g;
            b = (b < 0) ? 0 : b;

            dst.at<Vec3b>(k, j) = Vec3b((uchar)b,(uchar)g,(uchar)r);
        }
    }
}

void negativeY(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3);

    // Convert the YIQ image to a 3-channel image (Y, I, Q)
    Mat yiq_channels[3];
    split(src, yiq_channels);

    // Calculate the negative of the Y channel
    Mat negative_y = 255 - yiq_channels[0];

    // Merge the negative Y channel with the original I and Q channels
    Mat negative_yiq_channels[3] = { negative_y, yiq_channels[1], yiq_channels[2] };
    merge(negative_yiq_channels, 3, dst);
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

    for(int i = pi; i < (src.rows + pi); i++){
        for(int j = pj; j < (src.cols  + pj); j++){

            uchar r = src.at<Vec3b>(i - pi, j - pj)[2];
            uchar g = src.at<Vec3b>(i - pi, j - pj)[1];
            uchar b = src.at<Vec3b>(i - pi, j - pj)[0];

            dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
}

double dotProduct(vector<float>& mask, Mat& window){
    double result = 0;
    for(int i = 0; i < window.rows; i++){
        for(int j = 0; j < window.cols; j++){
            result += mask[i * window.cols + j] * window.at<Vec<float,1>>(i,j)[0];
        }
    }
    return result;
}

bool readMask(vector<float>& Mask, vector<int>& args, const char *path){
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
        cout << "Invalid Mask argsize" << endl;
        return false;
    }

    Mask.clear();

    for(int i = 1; i <= args.at(0); i++){
        stringstream mySstream(lines[i]);
        int j = 0;
        while(getline(mySstream, sstreamValue, ' ')){
            Mask.push_back(stof(sstreamValue));
            j++;
        }
        if(j != args.at(1)){
            Mask.clear();
            args.clear();
            return false;
        }
    }
    return true;
}

bool correlation_abs(Mat& src, Mat& dst, const char* path){
    vector<float> mask;
    vector<int> args;
    
    if(readMask(mask, args, path)){
        if(args.size() == 5){// 5 args are needed for the correlation, being the number of rows(m), number of columns(n), pivot row(pi), pivot cloumn(pj) and offset value.
            if(mask.size() == args.at(0) * args.at(1)){
                CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
                Mat paddedSrc;
                padImage(src, paddedSrc, args.at(0), args.at(1), args.at(2), args.at(3));
                dst.create(src.size(), CV_8UC3);

                Mat windowB, windowG, windowR;
                windowB.create(args.at(0), args.at(1), CV_32FC1);
                windowG.create(args.at(0), args.at(1), CV_32FC1);
                windowR.create(args.at(0), args.at(1), CV_32FC1);
                float r, g, b;

                int i = 0;
                int j = 0;

                while(i < src.rows){
                    j = 0;
                    while(j < src.cols){
                        for(int k = 0; k < args.at(0); k++){
                            for(int l = 0; l < args.at(1); l++){
                                windowB.at<Vec<float, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k, j + l)[0];//blue channel
                                windowG.at<Vec<float, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k, j + l)[1];//green channel
                                windowR.at<Vec<float, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k, j + l)[2];//red channel
                            }
                        }

                        b = dotProduct(mask, windowB);
                        g = dotProduct(mask, windowG);
                        r = dotProduct(mask, windowR);

                        b = (b < 0) ? -b : b;
                        g = (g < 0) ? -g : g;
                        r = (r < 0) ? -r : r;

                        b += args.at(4);
                        g += args.at(4);
                        r += args.at(4);

                        b = (b > 255) ? 255 : b;
                        g = (g > 255) ? 255 : g;
                        r = (r > 255) ? 255 : r;

                        dst.at<Vec3b>(i,j) = Vec3b((uchar)b, (uchar)g, (uchar)r);
                        
                        j++;
                    }
                    i++;
                }
                return true;
            } else
                cout << "Invalid mask size" << endl;
        } else
            cout << "Invalid arg size" << endl;
    }
    return false;
}

bool correlation_sat(Mat& src, Mat& dst, const char* path){
    vector<float> mask;
    vector<int> args;
    
    if(readMask(mask, args, path)){
        if(args.size() == 5){// 5 args are needed for the correlation, being the number of rows(m), number of columns(n), pivot row(pi), pivot cloumn(pj) and offset value.
            if(mask.size() == args.at(0) * args.at(1)){
                CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)
                Mat paddedSrc;
                padImage(src, paddedSrc, args.at(0), args.at(1), args.at(2), args.at(3));
                dst.create(src.size(), CV_8UC3);

                Mat windowB, windowG, windowR;
                windowB.create(args.at(0), args.at(1), CV_32FC1);
                windowG.create(args.at(0), args.at(1), CV_32FC1);
                windowR.create(args.at(0), args.at(1), CV_32FC1);
                float r, g, b;

                int i = 0;
                int j = 0;

                while(i < src.rows){
                    j = 0;
                    while(j < src.cols){
                        for(int k = 0; k < args.at(0); k++){
                            for(int l = 0; l < args.at(1); l++){
                                windowB.at<Vec<float, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k, j + l)[0];//blue channel
                                windowG.at<Vec<float, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k, j + l)[1];//green channel
                                windowR.at<Vec<float, 1>>(k, l) = paddedSrc.at<Vec3b>(i + k, j + l)[2];//red channel
                            }
                        }

                        // for(int k = 0; k < windowB.rows; k++){
                        //     for(int l = 0; l < windowB.cols; l++){
                        //         cout << "[" << windowB.at<Vec<double,1>>(k,l)[0] << "]";
                        //     }
                        //     cout << "\n";
                        // }

                        b = dotProduct(mask, windowB);
                        g = dotProduct(mask, windowG);
                        r = dotProduct(mask, windowR);

                        b = (b < 0) ? 0 : b;
                        g = (g < 0) ? 0 : g;
                        r = (r < 0) ? 0 : r;

                        b += args.at(4);
                        g += args.at(4);
                        r += args.at(4);

                        b = (b > 255) ? 255 : b;
                        g = (g > 255) ? 255 : g;
                        r = (r > 255) ? 255 : r;
                        
                        dst.at<Vec3b>(i,j) = Vec3b((uchar)b, (uchar)g, (uchar)r);
                        
                        j++;
                    }
                    i++;
                }
                return true;
            } else
                cout << "Invalid mask size" << endl;
        } else
            cout << "Invalid arg size" << endl;
    }
    return false;
}

void showImage(Mat& src, const char* name){
    imshow(name, src);
}

void histogramExpansion(Mat& src, Mat& dst){
    CV_Assert(src.channels() == 3); // Input image should have 3 channels (RGB)

    dst.create(src.size(), CV_8UC3);
    
    uchar rmax[4] = {0, 0, 0, 0};
    uchar rmin[4] = {255, 255, 255, 255};
    uchar red, green, blue;

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            red = src.at<Vec3b>(i,j)[2];
            green = src.at<Vec3b>(i,j)[1];
            blue = src.at<Vec3b>(i,j)[0];

            rmax[0] = (rmax[0] < red) ? red : rmax[0];
            rmin[0] = (rmin[0] > red) ? red : rmin[0];
            rmax[1] = (rmax[1] < green) ? green : rmax[1];
            rmin[1] = (rmin[1] > green) ? green : rmin[1];
            rmax[2] = (rmax[2] < blue) ? blue : rmax[2];
            rmin[2] = (rmin[2] > blue) ? blue : rmin[2];
        }
    }

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            red = src.at<Vec3b>(i,j)[2];
            green = src.at<Vec3b>(i,j)[1];
            blue = src.at<Vec3b>(i,j)[0];
            
            dst.at<Vec3b>(i,j) = Vec3b((uchar)round(255.0*(blue - rmin[2])/(rmax[2] - rmin[2])),
                                        (uchar)round(255.0*(green - rmin[1])/(rmax[1] - rmin[1])), 
                                        (uchar)round(255.0*(red - rmin[0])/(rmax[0] - rmin[0])));
        }
    }
}