#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "Connections.h"
using namespace std;

void test_alexnet(){
	// alexnet
	Connections cs;
	vector<string> t(
		{"conv0",
		"lrn0",
		"maxpool0",
		"conv1",
		"lrn1",
		"maxpool1",
		"conv2",
		"conv3",
		"conv4",
		"maxpool2",
		"fc0",
		"fc1",
		"fc2",
		"softmaxloss"});
	cs.add(t);
	cs.update();
	cs.show();
}
void test_vgg16(){
	// vgg
	Connections cs;
	vector<string> t(
		{"conv0",
		"conv1",
		"maxpool0",
		"conv2",
		"conv3",
		"maxpool1",
		"conv4",
		"conv5",
		"conv6",
		"maxpool2",
		"conv7",
		"conv8",
		"conv9",
		"maxpool3",
		"conv10",
		"conv11",
		"conv12",
		"maxpool4",
		"fc0",
		"fc1",
		"fc2",
		"softmaxloss"});
	cs.add(t);
	cs.update();
	cs.show();
}
void test_gnet_v1(){
	// gnet v1
	Connections cs;
	vector<string> t0({
			"conv0",
			"maxpool0",
			"lrn0",
			"conv1",
			"conv2",
			"lrn1",
			"maxpool1"});

	vector<vector<string> > t3a({
			{"maxpool1"},
				{"conv_3a_00"}, 
				{"conv_3a_10", "conv_3a_11"},
				{"conv_3a_20", "conv_3a_21"},
				{"maxpool_3a_30", "conv_3a_31"},
			{"concat_3a"}
		});
	vector<vector<string> > t3b({
			{"concat_3a"},
				{"conv_3b_00"}, 
				{"conv_3b_10", "conv_3b_11"},
				{"conv_3b_20", "conv_3b_21"},
				{"maxpool_3b_30", "conv_3b_31"},
			{"concat_3b"}
		});
	vector<string> t1({"concat_3b",
			"maxpool2"});
	vector<vector<string> > t4a({
			{"maxpool2"},
				{"conv_4a_00"}, 
				{"conv_4a_10", "conv_4a_11"},
				{"conv_4a_20", "conv_4a_21"},
				{"maxpool_4a_30", "conv_4a_31"},
			{"concat_4a"}
		});
	vector<vector<string> > t4b({
			{"concat_4a"},
				{"conv_4b_00"}, 
				{"conv_4b_10", "conv_4b_11"}, 
				{"conv_4b_20", "conv_4b_21"}, 
				{"maxpool_4b_30", "conv_4b_31"},
			{"concat_4b"}
		});
	vector<vector<string> > t4c({
			{"concat_4b"},
				{"conv_4c_00"}, 
				{"conv_4c_10", "conv_4c_11"},
				{"conv_4c_20", "conv_4c_21"}, 
				{"maxpool_4c_30", "conv_4c_31"},
			{"concat_4c"}
		});
	vector<vector<string> > t4d({
			{"concat_4c"},
				{"conv_4d_00"}, 
				{"conv_4d_10", "conv_4d_11"},
				{"conv_4d_20", "conv_4d_21"}, 
				{"maxpool_4d_30", "conv_4d_31"},
			{"concat_4d"},
		});
	vector<vector<string> > t4e({
			{"concat_4d"},
				{"conv_4e_00"}, 
				{"conv_4e_10", "conv_4e_11"},
				{"conv_4e_20", "conv_4e_21"}, 
				{"maxpool_4e_30", "conv_4e_31"},
			{"concat_4e"},
		});
	vector<string> t2({"concat_4e",
			"maxpool3"});
	vector<vector<string> > t5a({
			{"maxpool3"},
				{"conv_5a_00"}, 
				{"conv_5a_10", "conv_5a_11"}, 
				{"conv_5a_20", "conv_5a_21"}, 
				{"maxpool_5a_30", "conv_5a_31"},
			{"concat_5a"},
		});
	vector<vector<string> > t5b({
			{"concat_5a"},
				{"conv_5b_00"}, 
				{"conv_5b_10", "conv_5b_11"}, 
				{"conv_5b_20", "conv_5b_21"}, 
				{"maxpool_5b_30", "conv_5b_31"},
			{"concat_5b"},
		});
	vector<string> t3({"concat_5b",
			"avgpool0",
			"fc0",
			"softmaxloss"});
	cs.add(t0).
		add(t3a).add(t3b).
		add(t1).
		add(t4a).add(t4b).add(t4c).add(t4d).add(t4e).
		add(t2).
		add(t5a).add(t5b).
		add(t3);
	cs.update();
	cs.show();
}
void test_loop(){
	Connections cs;
	vector<string> t(
		{"a",
		"b" });
	vector<string> t1(
		{"b",
		"a" });
	cs.add(t).add(t1);
	cs.update();
	cs.show();
}
void test(){
	test_alexnet();
	test_vgg16();
	test_gnet_v1();
	test_loop();
}
int main(){
	test();

	return 0;
}
