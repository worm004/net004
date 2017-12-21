#include <string>
#include "NetGame.h"
using namespace std;
int main(int argc, char**argv){
	if(argc != 2){
		printf("./net_train path\n");
		return 0;
	}
	NetTrain t;
	t.load(argv[1]);
	t.init();
	t.run();
	return 0;
}
