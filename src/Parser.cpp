#include <vector>
#include <set>
#include "Parser.h"
#include "Net004.h"
#include "DataLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "LRNLayer.h"
#include "FCLayer.h"
#include "LossLayer.h"
#include "ConcatLayer.h"
#include "ActivityLayer.h"

using namespace std;
Parser::Parser(){
	read_net_funcs["conv"] = &Parser::read_net_conv;
	read_net_funcs["pool"] = &Parser::read_net_pool;
	read_net_funcs["activity"] = &Parser::read_net_activity;
	read_net_funcs["fc"] = &Parser::read_net_fc;
	read_net_funcs["loss"] = &Parser::read_net_loss;
	read_net_funcs["lrn"] = &Parser::read_net_lrn;
	read_net_funcs["data"] = &Parser::read_net_data;
	read_net_funcs["split"] = &Parser::read_net_split;
	read_net_funcs["concat"] = &Parser::read_net_concat;
}
void Parser::write(Net004* net, const std::string& net_path, const std::string& data_path){	
	Connections &cs = net->cs;
	Layers &ls = net->ls;
	ofstream net_file(net_path);

	net_file<<net->name<<endl;
	FILE* data_file = fopen(data_path.c_str(), "wb");
	int n = ls.count();
	for(int i=0;i<n;++i){
		Layer* layer = ls[cs.sorted_cs[i]];
		const string& layer_type = layer->type;
		const string& layer_name = layer->name;
		net_file<<"Layer: "<<layer_type<<" "<<layer_name<<endl;
		if(layer_type == "data"){
			write_net_data(layer, net_file);
			write_dat_data(layer, data_file);
		}
		else if(layer_type == "conv"){
			write_net_conv(layer, net_file);
			write_dat_conv(layer, data_file);
		}
		else if(layer_type == "pool"){
			write_net_pool(layer, net_file);
			write_dat_pool(layer, data_file);
		}
		else if(layer_type == "activity"){
			write_net_activity(layer, net_file);
			write_dat_activity(layer, data_file);
		}
		else if(layer_type == "fc"){
			write_net_fc(layer, net_file);
			write_dat_fc(layer, data_file);
		}
		else if(layer_type == "loss"){
			write_net_loss(layer, net_file);
			write_dat_loss(layer, data_file);
		}
		else{
			printf("no such parser for layer: %s\n",layer_type.c_str());
			exit(0);
		}
	}
	write_connections(net,net_file);
	net_file.close();
	fclose(data_file);
}

void Parser::write_net_data(Layer* layer, std::ofstream& ofile){
	// if need
	DataLayer *l = (DataLayer*)layer;
	ofile<<l->n<<" "<<l->c<<" "<<l->h<<" "<<l->w<<" "<<l->method<<endl;
}
void Parser::write_net_pool(Layer* layer, std::ofstream& ofile){
	PoolLayer *l = (PoolLayer*)layer;
	string method = l->get_method();
	ofile<<l->get_kernel()<<" "<<l->get_padding()<<" "<<l->get_stride()<<" "<<method<<endl;
}
void Parser::write_net_activity(Layer* layer, std::ofstream& ofile){
	ActivityLayer *l = (ActivityLayer*)layer;
	string method = l->get_method();
	ofile<<method<<endl;
}
void Parser::write_net_fc(Layer* layer, std::ofstream& ofile){
	FCLayer *l = (FCLayer*)layer;
	string activity = l->get_activity();
	if(activity.size()==0) activity = "none";
	ofile<<l->get_n()<<" "<<activity<<endl;
}
void Parser::write_net_loss(Layer* layer, std::ofstream& ofile){
	LossLayer *l = (LossLayer*)layer;
	string method = l->get_method();
	ofile<<method<<endl;
}
void Parser::write_net_conv(Layer* layer, std::ofstream& ofile){
	ConvLayer *l = (ConvLayer*)layer;
	string activity = l->get_activity();
	if(activity.size()==0) activity = "none";
	ofile<<l->get_kernel()<<" "<<l->get_filters()<<" "<<l->get_padding()<<" "<<l->get_stride()<<" "<<activity<<endl;
}
void Parser::write_connections(Net004* net, std::ofstream& ofile){
	Connections &cs = net->cs;
	Layers &ls = net->ls;
	ofile<<"Connections:"<<endl;

	const vector<string> &sorted_cs = cs.sorted_cs;
	int n = sorted_cs.size();
	for(int i=0;i<n;++i){
		const string &from = sorted_cs[i];
		if(!cs.exist(from)) continue;
		const set<string> &tos = cs[from];
		for(auto to: tos) ofile<<from<<" "<<to<<endl;
	}
}

void Parser::write_dat_data(Layer* layer, FILE* ofile){
}
void Parser::write_dat_conv(Layer* layer, FILE* ofile){
	ConvLayer* l = (ConvLayer*)layer;
	char buffer[100];
	sprintf(buffer,"Layer: %s weight",l->name.c_str());
	fwrite(buffer, sizeof(char), 100, ofile);
	fwrite(&(l->weight.n), sizeof(int), 1, ofile);
	fwrite(&(l->weight.c), sizeof(int), 1, ofile);
	fwrite(&(l->weight.h), sizeof(int), 1, ofile);
	fwrite(&(l->weight.w), sizeof(int), 1, ofile);
	fwrite(l->weight.data, sizeof(float), l->weight.nchw(), ofile);


	sprintf(buffer,"Layer: %s bias",l->name.c_str());
	fwrite(buffer, sizeof(char), 100, ofile);
	fwrite(&(l->bias.n), sizeof(int), 1, ofile);
	fwrite(&(l->bias.c), sizeof(int), 1, ofile);
	fwrite(&(l->bias.h), sizeof(int), 1, ofile);
	fwrite(&(l->bias.w), sizeof(int), 1, ofile);
	fwrite(l->bias.data, sizeof(float), l->bias.nchw(), ofile);
}
void Parser::write_dat_pool(Layer* layer, FILE* ofile){
}
void Parser::write_dat_activity(Layer* layer, FILE* ofile){
}
void Parser::write_dat_fc(Layer* layer, FILE* ofile){
	FCLayer* l = (FCLayer*)layer;
	char buffer[100];
	sprintf(buffer,"Layer: %s weight",l->name.c_str());
	fwrite(buffer, sizeof(char), 100, ofile);
	fwrite(&(l->weight.n), sizeof(int), 1, ofile);
	fwrite(&(l->weight.c), sizeof(int), 1, ofile);
	fwrite(&(l->weight.h), sizeof(int), 1, ofile);
	fwrite(&(l->weight.w), sizeof(int), 1, ofile);
	fwrite(l->weight.data, sizeof(float), l->weight.nchw(), ofile);

	sprintf(buffer,"Layer: %s bias",l->name.c_str());
	fwrite(buffer, sizeof(char), 100, ofile);
	fwrite(&(l->bias.n), sizeof(int), 1, ofile);
	fwrite(&(l->bias.c), sizeof(int), 1, ofile);
	fwrite(&(l->bias.h), sizeof(int), 1, ofile);
	fwrite(&(l->bias.w), sizeof(int), 1, ofile);
	fwrite(l->bias.data, sizeof(float), l->bias.nchw(), ofile);
}
void Parser::write_dat_loss(Layer* layer, FILE* ofile){
}


void Parser::read(const std::string& net_path, const std::string& data_path,Net004* net){
	read_net(net_path,net);
	net->check();
	net->setup();
	read_data(data_path,net);
	//net->show();
}
void Parser::read_net(const std::string& path, Net004* net){
	Layers & ls = net->ls;
	Connections& cs = net->cs;

	ifstream net_file(path);
	string line;
	char layer_type[100], layer_name[100];
	bool isconnect = false;
	getline(net_file,line);
	net->name = line;
	while(1){
		getline(net_file,line);
		if(net_file.eof()) break;
		if(line.find("Layer:") == 0) sscanf(line.c_str(),"Layer: %s %s",layer_type, layer_name);
		else if(line.find("Connections:") == 0) isconnect = true;
		else{
			if(isconnect){
				char from[100],to[100];
				sscanf(line.c_str(),"%s %s",from,to);
				cs.add(vector<string>({from,to}));
			}else if(read_net_funcs.find(layer_type) != read_net_funcs.end()){
				read_net_func func = read_net_funcs[layer_type];
				(this->*func)(line,layer_name,&ls);
			}
			else {
				printf("No such layer to parser %s\n",layer_type);
				exit(0);
			}
		}
	}
	net_file.close();
	cs.update();
}
void Parser::read_data(const std::string& path, Net004* net){
	Layers & ls = net->ls;
	FILE* file = fopen(path.c_str(), "rb");
	while(!feof(file)){
		char buffer[100],layer_name[100],data_name[100];
		fread(buffer, sizeof(char), 100, file);
		sscanf(buffer,"Layer: %s %s",layer_name,data_name);
		int n,c,h,w;
		fread(&n, sizeof(int), 1, file);
		fread(&c, sizeof(int), 1, file);
		fread(&h, sizeof(int), 1, file);
		fread(&w, sizeof(int), 1, file);
		int total = n*c*h*w;
		Layer* layer = ls[layer_name];
		Blob* b;
		if(layer->type == "conv"){
			ConvLayer* l = (ConvLayer*)layer;
			if(data_name == string("weight")) b = &(l->weight);
			else if(data_name == string("bias")) b = &(l->bias);
		}
		else if(layer->type == "fc"){
			FCLayer* l = (FCLayer*)layer;
			if(data_name == string("weight")) b = &(l->weight);
			else if(data_name == string("bias")) b = &(l->bias);
		}
		if((b->n != n) || (b->c != c) || (b->h != h) || (b->w != w)){
			printf("layer name: %s\nread: %d %d %d %d\n",layer_name,n,c,h,w);
			b->show();
			printf("Blob size does not match in net and data\n");
			exit(0);
		}
		fread(b->data,sizeof(float),total,file);
	}
	fclose(file);
}
void Parser::read_net_data(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	int n,c,h,w;
	sscanf(line.c_str(),"%d %d %d %d %s",&n,&c,&h,&w,method);
	if(batch_size != -1) n = batch_size;
	ls->add_data(name,n,c,h,w,method);
}
void Parser::read_net_split(const std::string& line, const std::string& name, Layers* ls){
	ls->add_split(name);
}
void Parser::read_net_lrn(const std::string& line, const std::string& name, Layers* ls){
	int local_size;
	float alpha,beta;
	sscanf(line.c_str(),"%d %f %f",&local_size,&alpha,&beta);
	ls->add_lrn(name,local_size,alpha,beta);
}
void Parser::read_net_conv(const std::string& line, const std::string& name, Layers* ls){
	char activity[100];
	int kernel, filters, padding, stride, group;
	sscanf(line.c_str(),"%d %d %d %d %d %s",&kernel, &filters, &padding, &stride, &group, activity);
	if (activity == string("none")) ls->add_conv(name,{filters,kernel,stride,padding,group},"");
	else ls->add_conv(name,{filters,kernel,stride,padding,group},activity);
}
void Parser::read_net_pool(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	int kernel, stride, padding;
	sscanf(line.c_str(),"%d %d %d %s",&kernel, &padding, &stride, method);
	ls->add_pool(name,{kernel, stride, padding},method);
}
void Parser::read_net_activity(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	sscanf(line.c_str(),"%s",method);
	ls->add_activity(name, method);
}
void Parser::read_net_fc(const std::string& line, const std::string& name, Layers* ls){
	char activity[100];
	int n;
	sscanf(line.c_str(),"%d %s",&n, activity);
	if (activity == string("none")) ls->add_fc(name,n,"");
	else ls->add_fc(name,n,activity);
}
void Parser::read_net_loss(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	sscanf(line.c_str(),"%s",method);
	ls->add_loss(name, method);
}
void Parser::read_net_concat(const std::string& line, const std::string& name, Layers* ls){
	char method[100];
	sscanf(line.c_str(),"%s",method);
	ls->add_concat(name, method);
}
