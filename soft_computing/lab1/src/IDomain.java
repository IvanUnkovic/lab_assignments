public interface IDomain extends Iterable<DomainElement>{

    public int getCardinality();
    public IDomain getComponent(int i);
    public int getNumberOfComponents();
    public int indexOfElement(DomainElement element);
    public DomainElement elementForIndex(int element);

}


