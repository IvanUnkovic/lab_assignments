
public abstract class Operations implements IUnaryFunction, IBinaryFunction{
    public static IFuzzySet unaryOperation(IFuzzySet set, IUnaryFunction fU){

        MutableFuzzySet newSet = new MutableFuzzySet(set.getDomain());
        int indexCounter = 0;
        for(DomainElement e : set.getDomain()){
            double value = set.getValueAt(e);
            newSet.getMemberships()[indexCounter] = fU.valueAt(value);
            indexCounter+=1;
        }
        return newSet;
    }

    public static IFuzzySet binaryOperation(IFuzzySet set1, IFuzzySet set2, IBinaryFunction fB){
        MutableFuzzySet newSet = new MutableFuzzySet(set1.getDomain());
        int indexCounter = 0;
        for (DomainElement e : set1.getDomain()){
            newSet.getMemberships()[indexCounter] = fB.valueAt(set1.getValueAt(e), set2.getValueAt(e));
            indexCounter+=1;
        }
        return newSet;
    }


    public static IUnaryFunction zadehNot() {
        return new IUnaryFunction() {
            @Override
            public double valueAt(double x) {
                return ((double) 1 - x);
            }
        };
    }

    public static IBinaryFunction zadehAnd() {
        return new IBinaryFunction() {
            @Override
            public double valueAt(double x, double y) {
                return Math.min(x,y);
            }
        };
    }

    public static IBinaryFunction zadehOr() {
        return new IBinaryFunction() {
            @Override
            public double valueAt(double x, double y) {
                return Math.max(x,y);
            }
        };
    }

    public static IBinaryFunction hamacherTNorm(double v) {
        return new IBinaryFunction() {
            @Override
            public double valueAt(double x, double y) {

                return (double) (x * y)/(v+(1-v)*(x+y-(double)x*y));
            }
        };
    }

    public static IBinaryFunction hamacherSNorm(double v) {
        return new IBinaryFunction() {
            @Override
            public double valueAt(double x, double y) {
                return (double)(x+y-(double)(2-v)*x*y)/(1-(double)(1-v)*x*y);
            }
        };
    }


}
