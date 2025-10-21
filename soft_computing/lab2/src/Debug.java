import java.util.Locale;

public abstract class Debug {
    public static void print(IDomain domain, String headingText) {
        if(headingText!=null) {
            System.out.println(headingText);
        }
        for(DomainElement e : domain) {
            System.out.println("Element domene: " + e);
        }
        System.out.println("Kardinalitet domene je: " + domain.getCardinality());
        System.out.println();
    }

    public static void printSet(IFuzzySet set, String headingText){
        if(headingText!=null) {
            System.out.println(headingText);
        }
        for (DomainElement e : set.getDomain()){
            double value = set.getValueAt(e);
            String formattedValue = String.format(Locale.US, "%.6f", value);
            System.out.println("d(" + e + ")=" + formattedValue);
        }
        System.out.println();
    }

}
