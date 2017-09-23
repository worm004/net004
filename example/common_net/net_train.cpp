#include <string>
#include "NetGame.h"
using namespace std;
int main(int argc, char**argv){
	NetTrain t;
	t.load("../train_settings/cifar01.txt");
	t.init();
	t.run();
	return 0;
}
