#include "ForwardTestRun.h"
using namespace std;
ForwardTestRun::ForwardTestRun(){}
ForwardTestRun::ForwardTestRun(const JsonValue& j): ForwardBackwardRun(j){
}
void ForwardTestRun::operator()(Net004& net, int cur) {
	if(cur%iter_interval != 0) return;
	printf("[%d]run: test\n",cur);
}
