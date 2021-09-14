#include <iostream>
#include <cmath>
#include <math.h>

using namespace std;

int main() {
    float s1[200],s2[200];
    float ref[5] = {-1,-1,1,1,1};

    // generate signal 1
    for(int i = 0; i<200;i++){
        s1[i] = 3*sin(i*M_PI/10+M_PI/2);
    }
    cout<<"signal 1 :"<<endl;
    
    for(auto &x:s1){
        cout<<x<<' ';
    }
    cout<<endl;
    // generate signal 2
    s2[0] = 1;
    for(int j =1; j<200;j++){
        s2[j] = ref[(j-1)%5];
    }


    cout<<"signal 2 :"<<endl;
    for(auto &y:s2){
        cout<<y<<' ';
    }

    //creat third signal
    float s3[200];
    for(int i=0; i<200;i++){
        s3[i] = s1[i]*s2[i];
    }
    cout<<endl;
    cout<<"signal 3 :"<<endl;
    for(auto &z:s3){
        cout<<z<<' ';
    }
    cout<<endl;
    //mean of the first quater and integral of second quater
    float sum=0;
    float mean;
    float integral=0;
    for(int i = 0;i<50;i++){
        sum+=s3[i];
    }
    mean = sum/50;
    cout<<"mean of first quarter of signal 3 is "<<mean<<endl;

    for(int i =50;i<100;i++){
        integral +=s3[i];
    }
    integral/=200;
    cout<<"integral of second quarter of signal is "<<integral<<endl;

return 0;

}