import java.util.Arrays;

public class DomainElement {
    private int[] values;

    public DomainElement(int[] values){
        this.values = values;
    }

    public int[] getValues() {
        return values;
    }

    public int getNumberOfComponents(){
        return values.length;
    }

    public int getComponentValue(int ind){
        return values[ind];
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof DomainElement)) return false;
        return Arrays.equals(this.values, ((DomainElement) obj).values);

    }

    public String toString(){
        if (this.getValues().length > 1){
            StringBuilder result = new StringBuilder("(");
            for (int i = 0; i < this.getValues().length; i++) {
                result.append(this.getValues()[i]);
                if (i < this.getValues().length - 1) {
                    result.append(", ");
                }
            }
            result.append(")");
            return result.toString();
        }else{
            return "" + this.getValues()[0];
        }
    }

    static public DomainElement of(int... valuesOfElement){
        int[] values = valuesOfElement;
        return new DomainElement(values);
    }
}
