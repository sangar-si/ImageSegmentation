
#include <torch/script.h> 
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

int main() {
    // Load model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("cpp_model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr<<"Error loading the model\n";
        return -1;
    }
    module.eval();  // Setting it to Eval mode
    std::cout<<"Model loaded successfully\n";

    // Load image
    cv::Mat img = cv::imread("test_img.jpg");
    if (img.empty()) {
        std::cerr<<"Could not open image!"<<std::endl;
        return -1;
    }
    std::cout<<"Image Loaded"<<std::endl;

    // Resize and preprocess the image to match training scenario. It is vital to match dimensions
    // similar to training in case of ML operations
    cv::resize(img, img, cv::Size(959, 640));

    // Preprocess
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255);

    // Convert to tensor
    torch::Tensor input_tensor = torch::from_blob(
        img.data,
        {1, img.rows, img.cols, 3},
        torch::kFloat32
    ).clone(); // Clone is important here as it creates new memory for the copied tensor. Process was getting "killed" without it. 

    input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous(); //Permute so that the dimensions match the input for the model. 
    std::cout<<"Input tensor size: "<<input_tensor.sizes()<<std::endl;

    // Forward pass. To perform inference in GPU, move it to GPU
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::Tensor pred = module.forward(inputs).toTensor();
    std::cout<<"Pred size: "<<pred.sizes()<<std::endl;


    // Processing the prediction, loading the tensor onto an opencv Mat
    auto result = pred[0].argmax(0).to(torch::kCPU, torch::kLong);
    std::cout<<"Result size: "<<result.sizes()<<std::endl;

    cv::Mat mask(result.size(0), result.size(1), CV_8U);
    //Load tensor onto Mat
    std::memcpy(mask.data, result.to(torch::kU8).contiguous().data_ptr(), rows * cols * sizeof(uint8_t));

    cv::Mat color_mask;
    cv::applyColorMap(mask * 20, color_mask, cv::COLORMAP_JET); 
    // Write it as jpeg
    cv::imwrite("output_mask.jpg", color_mask);
    return 0;
}