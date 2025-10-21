import java.util.Iterator;

public class MutableFuzzySet implements IFuzzySet{

    private double[] memberships;
    private IDomain domain;

    public MutableFuzzySet(IDomain d){
        domain = d;
        memberships = new double[d.getCardinality()];
        for (int i = 0; i < memberships.length; i++) {
            memberships[i] = 0;
        }
    }

    public IDomain getDomain(){
        return this.domain;
    }

    public double[] getMemberships(){ return this.memberships; }

    public double getValueAt(DomainElement e){
        if(domain.indexOfElement(e) == -1){
            return -1;
        }else {
            return memberships[domain.indexOfElement(e)];
        }
    }

    public MutableFuzzySet set(DomainElement e, double x){
        memberships[domain.indexOfElement(e)] = x;
        return this;
    }

}
