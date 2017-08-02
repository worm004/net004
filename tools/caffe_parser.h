#ifndef CAFFE_PARSER_H
#define CAFFE_PARSER_H
#include <string>
#include <vector>
#include <fstream>
#include "caffe/caffe.hpp"

class CaffeModelParser{
public:
	void load_caffe_model(const std::string& net_path, const std::string& model_path);
	void write(const std::string& net_path, const std::string& model_path);
	void show_layers();

private:
	void write_net_conv(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile);
	void write_net_pool(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile);
	void write_net_data(const std::string& layer_name, const std::string& blob_name, int n,int c,int h,int w, std::ofstream& ofile);
	void write_net_relu(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile);
	void write_net_fc(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile);
	void write_net_softmaxloss(const std::string& layer_name, const caffe::LayerParameter& param, std::ofstream& ofile);
	void write_net(const std::string& net_path);
	void write_model(const std::string& model_path);
	void read_connections();
private:
	std::shared_ptr<caffe::Net<float> > net;
	std::vector<std::pair<std::string, std::string> > connections;
};
#endif
