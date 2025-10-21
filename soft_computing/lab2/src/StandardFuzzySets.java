public abstract class StandardFuzzySets {

    public static IIntUnaryFunction lFunction(int i1, int i2){
        return new IIntUnaryFunction() {

            @Override
            public double valueAt(int x){
                if(x<i1){
                    return 1;
                }else if(i1 <= x && x < i2){
                    return (double) (i2-x)/(i2-i1);
                }else{
                    return 0;
                }
            }
        };
    }

    public static IIntUnaryFunction gammaFunction(int i1, int i2){

        return new IIntUnaryFunction() {

            @Override
            public double valueAt(int x) {
                if(x < i1){
                    return 0;
                }else if(i1 <= x && x < i2){
                    return (double) (x-i1)/(i2-i1);
                }else{
                    return 1;
                }
            }
        };

    }

    public static IIntUnaryFunction lambdaFunction(int i1, int i2, int i3){
        return new IIntUnaryFunction() {
            @Override
            public double valueAt(int x) {
                if (x < i1){
                    return 0;
                }else if(i1 <= x && x < i2){
                    return (double) (x-i1)/(i2-i1);
                }else if(i2 <= x && x < i3){
                    return (double) (i3-x)/(i3-i2);
                }else{
                    return 0;
                }
            }
        };

    }
}
