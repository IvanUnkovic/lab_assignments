import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public abstract class Relations {

    public static boolean isUtimesURelation(IFuzzySet relation){

        if (relation.getDomain() instanceof SimpleDomain){
            return false;
        }

        if(relation.getDomain().getNumberOfComponents()==2 && relation.getDomain().getComponent(0) == relation.getDomain().getComponent(1)){
            return true;
        }

        return false;
    }

    public static boolean isSymmetric(IFuzzySet relation){

        if(!isUtimesURelation(relation)){
            return false;
        }

        for (DomainElement e : relation.getDomain()){
            int[] domainElementValues = e.getValues();
            int[] otherValues = new int[domainElementValues.length];

            otherValues[0] = domainElementValues[1];
            otherValues[1] = domainElementValues[0];

            if (Arrays.equals(domainElementValues, otherValues)){
                continue;
            }

            if (relation.getValueAt(e) != relation.getValueAt(DomainElement.of(otherValues))){
                return false;
            }
        }

        return true;
    }
    public static boolean isReflexive(IFuzzySet relation){

        if(!isUtimesURelation(relation)){
            return false;
        }

        for (DomainElement e : relation.getDomain()){
            if(e.getValues()[0] == e.getValues()[1] && relation.getValueAt(e) != 1){
                return false;
            }
        }

        return true;
    }

    public static boolean isMaxMinTransitive(IFuzzySet relation){

        if(!isUtimesURelation(relation)){
            return false;
        }

        for (DomainElement e : relation.getDomain()){
            Map<DomainElement, Double> minMap = new HashMap<>();
            for(DomainElement f : relation.getDomain().getComponent(0)){
                DomainElement d1 = DomainElement.of(e.getValues()[0],Integer.parseInt(f.toString()));
                DomainElement d2 = DomainElement.of(Integer.parseInt(f.toString()), e.getValues()[1]);
                double minimum = Math.min(relation.getValueAt(d1), relation.getValueAt(d2));
                minMap.put(f, minimum);
            }

            int[] dummy = new int[1];
            double maximum = Double.MIN_VALUE;
            DomainElement maxDomainElement = new DomainElement(dummy);
            for (Map.Entry<DomainElement, Double> entry : minMap.entrySet()) {
                double value = entry.getValue();
                if (value > maximum) {
                    maximum = value;
                    maxDomainElement = entry.getKey();
                }
            }

            if(relation.getValueAt(e) < Integer.parseInt(maxDomainElement.toString())){
                return false;
            }
        }
        return true;
    }

    public static IFuzzySet compositionOfBinaryRelations(IFuzzySet r1, IFuzzySet r2){
        SimpleDomain[] domains = {(SimpleDomain) r1.getDomain().getComponent(0), (SimpleDomain) r2.getDomain().getComponent(1)};
        CompositeDomain d = new CompositeDomain(domains);
        IFuzzySet newR = new MutableFuzzySet(d);

        for(int i = domains[0].getFirst(); i<domains[0].getLast(); i++){
            for(int j = domains[1].getFirst(); j<domains[1].getLast(); j++){

            }
        }


        return newR;
    }

    public static boolean isFuzzyEquivalence(IFuzzySet r){
        return isReflexive(r) && isSymmetric(r) && isMaxMinTransitive(r);
    }


}
