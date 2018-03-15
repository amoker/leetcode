import java.util.PriorityQueue;

public class MedianFinder {
    
    private PriorityQueue<Integer> pqUp;
    private PriorityQueue<Integer> pqDown;

    /** initialize your data structure here. */
    public MedianFinder() {
        this.pqDown = new PriorityQueue<>();
        this.pqUp = new PriorityQueue<>((x, y) -> (y - x));
    }
    
    public void addNum(int num) {
        if(pqUp.size()>pqDown.size()){
        	if(pqUp.peek()>num){
        		pqDown.add(pqUp.poll());
        		pqUp.add(num);
        	}
        	else{
        		pqDown.add(num);
        	}
        }
        else{
        	if(pqUp.size()==0) pqUp.add(num);
        	else if(pqDown.peek()<num){
        		pqUp.add(pqDown.poll());
        		pqDown.add(num);
        	}
        	else{
        		pqUp.add(num);
        	}
        }
    }
    
    public double findMedian() {
        if(pqUp.size() == pqDown.size() ){
        	return pqUp.peek()+(pqDown.peek()-pqUp.peek())/2.0;
        }
        else return pqUp.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */