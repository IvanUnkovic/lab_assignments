import java.util.Iterator;
import java.util.Spliterator;
import java.util.function.Consumer;

public abstract class Domain implements IDomain {
    static public IDomain intRange(int r1, int r2){
        return new SimpleDomain(r1, r2);
    }

    static public Domain combine(IDomain d1, IDomain d2){
        SimpleDomain simpleDomain1 = (SimpleDomain) d1;
        SimpleDomain simpleDomain2 = (SimpleDomain) d2;
        SimpleDomain[] domains = new SimpleDomain[]{simpleDomain1, simpleDomain2};
        return new CompositeDomain(domains);
    }

    public abstract int indexOfElement(DomainElement element);

    public abstract DomainElement elementForIndex(int index);

}
