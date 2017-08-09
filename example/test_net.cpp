#include "Net004.h"
using namespace std;
void test_cifar10(){
	Net004 net("cifar10");
	Layers & ls = net.ls;
	ls.add_data("data",1,3,32,32,"image");
	ls.add_data("gt",1,1,1,1,"label");
	ls.add_conv("conv0",{32,5,1,2,1},1,"relu");
	ls.add_pool("maxpool0",{3,2,0},"max");
	ls.add_conv("conv1",{32,5,1,2,1},1,"relu");
	ls.add_pool("maxpool1",{3,2,0},"max");
	ls.add_conv("conv2",{64,5,1,2,1},1,"relu");
	ls.add_pool("maxpool2",{3,2,0},"avg");
	ls.add_fc("fc0",64,1,"");
	ls.add_fc("fc1",10,1,"");
	ls.add_loss("softmaxloss","softmax");
	//ls.show();
	Connections& cs = net.cs;
	vector<string> t0({
			"data",
			"conv0",
			"maxpool0",
			"conv1",
			"maxpool1",
			"conv2",
			"maxpool2",
			"fc0",
			"fc1",
			"softmaxloss"});
	vector<string> t1({"gt","softmaxloss"});
	cs.add(t0).add(t1);
	cs.update();
	//cs.show();
	
	net.check();
	net.setup();
	net.show();
}
void test_concat(){
	Net004 net("test_concat");
	Layers & ls = net.ls;
	ls.add_data("data",1,3,227,227,"image");
	ls.add_data("gt",1,1,1,1,"label");
	ls.add_conv("conv0",{96,11,4,0,1},1,"relu");
	ls.add_conv("conv_3a_00",{64,1,1,0,1},1,"relu");
	ls.add_conv("conv_3a_10",{96,1,1,0,1},1,"relu");
	ls.add_conv("conv_3a_11",{128,3,1,1,1},1,"relu");
	ls.add_conv("conv_3a_20",{16,1,1,0,1},1,"relu");
	ls.add_conv("conv_3a_21",{32,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_3a_30",{3,1,1},"max");
	ls.add_conv("conv_3a_31",{32,1,1,0,1},1,"relu");
	ls.add_concat("concat_3a");
	ls.add_fc("fc0",1000,1,"");
	ls.add_loss("softmaxloss","softmax");

	Connections& cs = net.cs;
	vector<string> t0({
			"data",
			"conv0"});
	vector<vector<string> > t1a({
			{"conv0"},
				{"conv_3a_00"}, 
				{"conv_3a_10", "conv_3a_11"},
				{"conv_3a_20", "conv_3a_21"},
				{"maxpool_3a_30", "conv_3a_31"},
			{"concat_3a"}
		});
	vector<string> t2({"concat_3a",
			"fc0",
			"softmaxloss"});
	vector<string> t3({"gt","softmaxloss"});
	cs.add(t0).add(t1a).add(t2).add(t3);
	cs.update();

	// net
	net.check();
	net.setup();
	net.show();
}

void test_alexnet(){
	Net004 net("alexnet");

	// layers
	Layers &ls = net.ls;
	ls.add_data("data",1,3,227,227,"image");
	ls.add_data("gt",1,1,1,1,"label");
	ls.add_conv("conv0",{96,11,4,0,1},1,"relu");
	ls.add_lrn("lrn0",5,0.0001,0.75);
	ls.add_pool("maxpool0",{3,2,0},"max");
	ls.add_conv("conv1",{256,5,1,2,1},1,"relu");
	ls.add_lrn("lrn1",5,0.0001,0.75);
	ls.add_pool("maxpool1",{3,2,0},"max");
	ls.add_conv("conv2",{384,3,1,1,1},1,"relu");
	ls.add_conv("conv3",{384,3,1,1,1},1,"relu");
	ls.add_conv("conv4",{256,3,1,1,1},1,"relu");
	ls.add_pool("maxpool2",{3,2,0},"max");
	ls.add_fc("fc0",4096,1,"relu");
	ls.add_fc("fc1",4096,1,"relu");
	ls.add_fc("fc2",1000,1,"relu");
	ls.add_loss("softmaxloss","softmax");

	// connections
	Connections& cs = net.cs;
	vector<string> t(
		{"data",
		"conv0",
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
	vector<string> t2({"gt","softmaxloss"});
	cs.add(t).add(t2);
	cs.update();

	// net
	net.check();
	net.setup();
	net.show();
}
void test_vgg16(){

	Net004 net("vgg16");

	// layers
	Layers &ls=net.ls;
	ls.add_data("data",1,3,224,224,"image");
	ls.add_data("gt",1,1,1,1,"label");
	ls.add_conv("conv0",{64,3,1,1,1},1,"relu");
	ls.add_conv("conv1",{64,3,1,1,1},1,"relu");
	ls.add_pool("maxpool0",{2,2,0},"max");
	ls.add_conv("conv2",{128,3,1,1,1},1,"relu");
	ls.add_conv("conv3",{128,3,1,1,1},1,"relu");
	ls.add_pool("maxpool1",{2,2,0},"max");
	ls.add_conv("conv4",{256,3,1,1,1},1,"relu");
	ls.add_conv("conv5",{256,3,1,1,1},1,"relu");
	ls.add_conv("conv6",{256,3,1,1,1},1,"relu");
	ls.add_pool("maxpool2",{2,2,0},"max");
	ls.add_conv("conv7",{512,3,1,1,1},1,"relu");
	ls.add_conv("conv8",{512,3,1,1,1},1,"relu");
	ls.add_conv("conv9",{512,3,1,1,1},1,"relu");
	ls.add_pool("maxpool3",{2,2,0},"max");
	ls.add_conv("conv10",{512,3,1,1,1},1,"relu");
	ls.add_conv("conv11",{512,3,1,1,1},1,"relu");
	ls.add_conv("conv12",{512,3,1,1,1},1,"relu");
	ls.add_pool("maxpool4",{2,2,0},"max");
	ls.add_fc("fc0",4096,1,"relu");
	ls.add_fc("fc1",4096,1,"relu");
	ls.add_fc("fc2",1000,1,"relu");
	ls.add_loss("softmaxloss","softmax");

	// connections
	Connections &cs=net.cs;
	vector<string> t(
		{"data",
		"conv0",
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
	vector<string> t2({"gt","softmaxloss"});
	cs.add(t).add(t2);
	cs.update();

	// net
	net.check();
	net.setup();
	net.show();
}
void test_gnet_v1(){
	Net004 net("gnet_v1");

	// layers
	Layers &ls = net.ls;
	ls.add_data("data",1,3,224,224,"image");
	ls.add_data("gt",1,1,1,1,"label");

	ls.add_conv("conv0",{64,7,2,3,1},1,"relu");
	ls.add_pool("maxpool0",{3,2,0},"max");
	ls.add_lrn("lrn0",5,0.0001,0.75);
	ls.add_conv("conv1",{64,1,1,0,1},1,"relu");
	ls.add_conv("conv2",{192,3,1,1,1},1,"relu");
	ls.add_lrn("lrn1",5,0.0001,0.75);
	ls.add_pool("maxpool1",{3,2,0},"max");

	//3a
	ls.add_conv("conv_3a_00",{64,1,1,0,1},1,"relu");
	ls.add_conv("conv_3a_10",{96,1,1,0,1},1,"relu");
	ls.add_conv("conv_3a_11",{128,3,1,1,1},1,"relu");
	ls.add_conv("conv_3a_20",{16,1,1,0,1},1,"relu");
	ls.add_conv("conv_3a_21",{32,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_3a_30",{3,1,1},"max");
	ls.add_conv("conv_3a_31",{32,1,1,0,1},1,"relu");
	ls.add_concat("concat_3a");
	
	//3b
	ls.add_conv("conv_3b_00",{128,1,1,0,1},1,"relu");
	ls.add_conv("conv_3b_10",{128,1,1,0,1},1,"relu");
	ls.add_conv("conv_3b_11",{192,3,1,1,1},1,"relu");
	ls.add_conv("conv_3b_20",{32,1,1,0,1},1,"relu");
	ls.add_conv("conv_3b_21",{96,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_3b_30",{3,1,1},"max");
	ls.add_conv("conv_3b_31",{64,1,1,0,1},1,"relu");
	ls.add_concat("concat_3b");

	ls.add_pool("maxpool2",{3,2,0},"max");

	//4a
	ls.add_conv("conv_4a_00",{192,1,1,0,1},1,"relu");
	ls.add_conv("conv_4a_10",{96,1,1,0,1},1,"relu");
	ls.add_conv("conv_4a_11",{208,3,1,1,1},1,"relu");
	ls.add_conv("conv_4a_20",{16,1,1,0,1},1,"relu");
	ls.add_conv("conv_4a_21",{48,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_4a_30",{3,1,1},"max");
	ls.add_conv("conv_4a_31",{64,1,1,0,1},1,"relu");
	ls.add_concat("concat_4a");

	//4b
	ls.add_conv("conv_4b_00",{160,1,1,0,1},1,"relu");
	ls.add_conv("conv_4b_10",{112,1,1,0,1},1,"relu");
	ls.add_conv("conv_4b_11",{224,3,1,1,1},1,"relu");
	ls.add_conv("conv_4b_20",{24,1,1,0,1},1,"relu");
	ls.add_conv("conv_4b_21",{64,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_4b_30",{3,1,1},"max");
	ls.add_conv("conv_4b_31",{64,1,1,0,1},1,"relu");
	ls.add_concat("concat_4b");

	//4c
	ls.add_conv("conv_4c_00",{128,1,1,0,1},1,"relu");
	ls.add_conv("conv_4c_10",{128,1,1,0,1},1,"relu");
	ls.add_conv("conv_4c_11",{256,3,1,1,1},1,"relu");
	ls.add_conv("conv_4c_20",{24,1,1,0,1},1,"relu");
	ls.add_conv("conv_4c_21",{64,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_4c_30",{3,1,1},"max");
	ls.add_conv("conv_4c_31",{64,1,1,0,1},1,"relu");
	ls.add_concat("concat_4c");

	//4d
	ls.add_conv("conv_4d_00",{112,1,1,0,1},1,"relu");
	ls.add_conv("conv_4d_10",{144,1,1,0,1},1,"relu");
	ls.add_conv("conv_4d_11",{288,3,1,1,1},1,"relu");
	ls.add_conv("conv_4d_20",{32,1,1,0,1},1,"relu");
	ls.add_conv("conv_4d_21",{64,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_4d_30",{3,1,1},"max");
	ls.add_conv("conv_4d_31",{64,1,1,0,1},1,"relu");
	ls.add_concat("concat_4d");

	//4e
	ls.add_conv("conv_4e_00",{256,1,1,0,1},1,"relu");
	ls.add_conv("conv_4e_10",{160,1,1,0,1},1,"relu");
	ls.add_conv("conv_4e_11",{320,3,1,1,1},1,"relu");
	ls.add_conv("conv_4e_20",{32,1,1,0,1},1,"relu");
	ls.add_conv("conv_4e_21",{128,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_4e_30",{3,1,1},"max");
	ls.add_conv("conv_4e_31",{128,1,1,0,1},1,"relu");
	ls.add_concat("concat_4e");

	ls.add_pool("maxpool3",{3,2,0},"max");

	//5a
	ls.add_conv("conv_5a_00",{256,1,1,0,1},1,"relu");
	ls.add_conv("conv_5a_10",{160,1,1,0,1},1,"relu");
	ls.add_conv("conv_5a_11",{320,3,1,1,1},1,"relu");
	ls.add_conv("conv_5a_20",{32,1,1,0,1},1,"relu");
	ls.add_conv("conv_5a_21",{128,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_5a_30",{3,1,1},"max");
	ls.add_conv("conv_5a_31",{128,1,1,0,1},1,"relu");
	ls.add_concat("concat_5a");

	//5b
	ls.add_conv("conv_5b_00",{384,1,1,0,1},1,"relu");
	ls.add_conv("conv_5b_10",{192,1,1,0,1},1,"relu");
	ls.add_conv("conv_5b_11",{384,3,1,1,1},1,"relu");
	ls.add_conv("conv_5b_20",{48,1,1,0,1},1,"relu");
	ls.add_conv("conv_5b_21",{128,5,1,2,1},1,"relu");
	ls.add_pool("maxpool_5b_30",{3,1,1},"max");
	ls.add_conv("conv_5b_31",{128,1,1,0,1},1,"relu");
	ls.add_concat("concat_5b");

	ls.add_pool("avgpool0",{7,1,0},"avg");
	ls.add_fc("fc0",1000,1,"");
	ls.add_loss("softmaxloss","softmax");

	// connections
	Connections &cs = net.cs;
	vector<string> t0({
			"data",
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
	vector<string> t4({"gt","softmaxloss"});
	cs.add(t0).
		add(t3a).add(t3b).
		add(t1).
		add(t4a).add(t4b).add(t4c).add(t4d).add(t4e).
		add(t2).
		add(t5a).add(t5b).
		add(t3).add(t4);
	cs.update();

	// net
	net.check();
	net.setup();
	net.show();
}
void test(){
	test_cifar10();
	//test_alexnet();
	//test_vgg16();
	//test_concat();
	//test_gnet_v1();
}
int main(){
	test();
	return 0;
}
