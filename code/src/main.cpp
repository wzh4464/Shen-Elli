#include "EllipseDetector.h"
#include "util.h"
#include "EdgeDetector.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main()
{
    EllipseDetector ellipse_Detector;
    std::vector<Ellipse> ellipses;
    string input_dir = "/home/zihan/dataset/test-img/";
    string output_dir = "/home/zihan/dataset/test-img/Shen/";

    // 确保输出目录存在
    fs::create_directories(output_dir);

    for (const auto & entry : fs::directory_iterator(input_dir))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            auto start = clock();
            string input_path = entry.path().string();
            cv::Mat img = cv::imread(input_path);
            
            if (img.empty())
            {
                std::cout << "无法读取图片: " << input_path << std::endl;
                continue;
            }

            ellipses = ellipse_Detector.DetectImage(img);
            cv::Mat3b img0 = ellipse_Detector.image();
            draw_ellipses_all(ellipses, img0);

            string output_path = output_dir + entry.path().filename().string();
            cv::imwrite(output_path, img0);

            double time_spent = (clock() - start) * 1000.0 / CLOCKS_PER_SEC;
            printf("处理 %s 用时: %.2f ms\n", entry.path().filename().string().c_str(), time_spent);
            printf("Number of ellipses: %zu\n", ellipses.size());  // 添加这行来输出椭圆数量
        }
    }

    std::cout << "所有图片处理完成，结果保存在 " << output_dir << std::endl;
    return 0;
}