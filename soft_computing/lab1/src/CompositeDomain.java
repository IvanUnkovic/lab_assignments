import java.util.Iterator;

public class CompositeDomain extends Domain{

    private SimpleDomain[] components;

    public CompositeDomain(SimpleDomain[] domain){
        components = domain;
    }

    @Override
    public int getCardinality() {
        int cardinality = 1;
        for (SimpleDomain component : components) {
            cardinality *= component.getCardinality();
        }
        return cardinality;
    }

    @Override
    public IDomain getComponent(int i){
        return components[i];
    }

    @Override
    public int getNumberOfComponents() {
        return components.length;
    }

    @Override
    public int indexOfElement(DomainElement element) {
        int indexCounter = 0;
        for(DomainElement e : this) {
            if(e.equals(element)){
                return indexCounter;
            }
            indexCounter +=1;
        }
        return -1;
    }

    @Override
    public DomainElement elementForIndex(int index) {
        int indexCounter = 0;
        for(DomainElement e: this){
            if(indexCounter == index){
                return e;
            }
            indexCounter+=1;
        }
        return null;
    }

    @Override
    public Iterator<DomainElement> iterator() {
        return new CDIterator();
    }

    private class CDIterator implements Iterator<DomainElement> {
        private int[] iterator;

        public CDIterator() {
            iterator = new int[components.length];
            for(int i = 0; i < components.length ; i++) {
                iterator[i] = components[i].getFirst();
            }
            iterator[iterator.length-1]--;
        }

        @Override
        public boolean hasNext() {
            for(int i = 0; i < iterator.length ; i++) {
                if(iterator[i] != components[i].getLast()-1)
                    return true;
            }
            return false;
        }

        @Override
        public DomainElement next() {
            boolean flag = true;
            for(int i = iterator.length-1; i >= 0 ; i--) {
                if(flag) {
                    if(iterator[i] == components[i].getLast()-1) {
                        flag = true;
                        iterator[i] = components[i].getFirst();
                    } else {
                        flag = false;
                        iterator[i]++;
                    }
                } else{
                    break;
                }
            }
            return new DomainElement(iterator);
        }
    }


}
