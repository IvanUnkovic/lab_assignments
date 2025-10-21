import java.util.Iterator;
import java.util.NoSuchElementException;

public class SimpleDomain extends Domain {
    private int first;
    private int last;

    public SimpleDomain(int first, int last){
        this.first = first;
        this.last = last;
    }

    @Override
    public int getCardinality() {
        return this.last - this.first;
    }

    @Override
    public IDomain getComponent(int i) {
        return this;
    }

    @Override
    public int getNumberOfComponents(){
        return 1;
    }

    @Override
    public int indexOfElement(DomainElement element){
        for(int i=this.first; i<this.last; i++){
            if (element.getValues()[0]==i){
                return i-this.first;
            }
        }
        return -1;
    }

    @Override
    public DomainElement elementForIndex(int index) {
        int value = this.first + index;
        return new DomainElement(new int[]{value});
    }

    public int getFirst(){
        return this.first;
    }

    public int getLast(){
        return this.last;
    }

    @Override
    public Iterator<DomainElement> iterator() {
        return new SDIterator();
    }

    private class SDIterator implements Iterator<DomainElement> {
        private int iterator = getFirst();

        @Override
        public boolean hasNext(){
            return iterator < getLast();
        }

        @Override
        public DomainElement next(){
            if(!hasNext()){
                throw new NoSuchElementException();
            }else{
                return new DomainElement(new int[]{iterator++});
            }
        }
    }
}
