public class CalculatedFuzzySet implements IFuzzySet{

    private IDomain d;
    IIntUnaryFunction f;

    public CalculatedFuzzySet(IDomain d, IIntUnaryFunction f){
        this.d = d;
        this.f = f;
    }

    public IDomain getDomain(){
        return this.d;

    }

    public double getValueAt(DomainElement e){
        return f.valueAt(d.indexOfElement(e));    }
}
