#include <string>
#include "JsonParser.h"
//#include "NetGame.h"
using namespace std;
int main(int argc, char**argv){
	JsonParser j;
	//j.read("../models/cifar.net004.net");
	j.read("../train_settings/cifar01.txt");
	j.show();
	//j.write("test.net");
	//NetTrain t;
	//t.load("../trian_settings/cifar01.txt");
	//t.init();
	//t.run();
	return 0;
}
