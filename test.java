import java.math.BigInteger;
import java.util.*;


public class test {
	public static int islandPerimeter(int[][] grid) {
		int stripes, sum = 0;
		int x = grid.length;
		int y = grid[0].length;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (grid[i][j] == 0) {
					continue;
				}
				stripes = 4;
				if (i != 0) {
					if (grid[i - 1][j] == 1)
						stripes--;
				}
				if (i != (x - 1)) {
					if (grid[i + 1][j] == 1)
						stripes--;
				}
				if (j != 0) {
					if (grid[i][j - 1] == 1)
						stripes--;
				}
				if (j != (y - 1)) {
					if (grid[i][j + 1] == 1)
						stripes--;
				}
				sum += stripes;
			}
		}
		return sum;
	}

	public String convertToBase7(int num) {
		List<String> list = new ArrayList<String>();
		if (num < 0) {
			num *= -1;
			list.add("-");
		}
		while (num / 7 != 0) {
			list.add(num % 7 + "");
			num /= 7;
		}
		return list.toString();
	}

	public int numberOfBoomerangs(int[][] points) {
		int res = 0;

		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < points.length; i++) {
			for (int j = 0; j < points.length; j++) {
				if (i == j)
					continue;

				int d = getDistance(points[i], points[j]);
				map.put(d, map.getOrDefault(d, 0) + 1);
			}

			for (int val : map.values()) {
				res += val * (val - 1);
			}
			map.clear();

		}

		return res;
	}

	private int getDistance(int[] a, int[] b) {
		int dx = a[0] - b[0];
		int dy = a[1] - b[1];

		return dx * dx + dy * dy;
	}

	public TreeNode sortedArrayToBST(int[] nums) {
		if (nums == null)
			return null;
		TreeNode root = new TreeNode(nums[0]);
		root.left = build(nums, 1);
		root.right = build(nums, 2);
		return root;
	}

	public TreeNode build(int[] nums, int i) {
		if (i < nums.length) {
			TreeNode root = new TreeNode(nums[i]);
			root.left = build(nums, 2 * i);
			root.right = build(nums, 2 * i + 1);
			return root;
		}
		return null;
	}

	public static int maxProfit(int[] prices) {
		if (prices.length <= 1)
			return 0;
		int[] dif = new int[prices.length - 1];
		for (int i = 0; i < dif.length; i++)
			dif[i] = prices[i + 1] - prices[i];
		return subArray(dif, 0, dif.length - 1);
	}

	public static int subArray(int[] dif, int low, int high) {
		if (low == high)
			return dif[low];
		int mid = (low + high) / 2;
		int a = Math.max(subArray(dif, low, mid), subArray(dif, mid + 1, high));
		int b = Math.max(crossingSubArray(dif, low, mid, high), 0);
		if (a > b)
			a++;
		return Math.max(a, b);
	}

	public static int crossingSubArray(int[] dif, int low, int mid, int high) {
		int leftMax = dif[mid];
		int rightMax = dif[mid];
		for (int sum = 0, i = mid; i >= low; i--) {
			sum += dif[i];
			if (leftMax < sum)
				leftMax = sum;
		}
		for (int sum = 0, i = mid; i <= high; i++) {
			sum += dif[i];
			if (rightMax < sum)
				rightMax = sum;
		}
		return leftMax + rightMax - dif[mid];

	}

	public boolean isPowerOfFour(int num) {
		int b = 0xaaaaaaab;
		int a = (num & b);
		return (num > 0 && Integer.bitCount(num) == 1 && a == 0);
	}

	Map<Integer, Integer> map;
	int max;

	public int[] findMode(TreeNode root) {
		if (root == null)
			return null;
		this.map = new HashMap<>();
		List<Integer> list = new ArrayList<>();

		inOrder(root);

		for (int key : map.keySet()) {
			if (map.get(key) == max)
				list.add(key);
		}

		int[] result = new int[list.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;

	}

	public boolean isHappy(int n) {
		Set<Integer> set = new HashSet<>();
		do {
			int num = 0;
			while (n != 0) {
				num += Math.pow(n % 10, 2);
				n /= 10;
			}
			n = num;
		} while (set.add(n) && n != 1);
		return n == 1;
	}

	public int pathSum(TreeNode root, int sum) {
		this.map = new HashMap<>();
		inOrder(root);
		return map.getOrDefault(sum, 0);
	}

	private void inOrder(TreeNode node) {
		if (node == null)
			return;
		int total = 0;
		add(node, total);
		inOrder(node.left);
		inOrder(node.right);
	}

	private void add(TreeNode node, int total) {
		if (node == null)
			return;
		total += node.val;
		map.put(total, map.getOrDefault(total, 0) + 1);
		add(node.left, total);
		add(node.right, total);
	}

	public boolean detectCapitalUse(String word) {
		if (word.length() == 0)
			return false;
		char[] words = word.toCharArray();
		if (Character.isUpperCase(words[0])) {
			for (int i = 2; i < words.length; i++) {
				if (Character.isUpperCase(words[i]) != Character.isUpperCase(words[1]))
					return false;
			}
		} else {
			for (char c : words) {
				if (!Character.isLowerCase(c))
					return false;
			}
		}
		return true;
	}

	public List<List<Integer>> generate(int numRows) {
		List<List<Integer>> res = new LinkedList<>();
		for (int i = 0; i < numRows; i++) {
			List<Integer> list = new LinkedList<>();
			list.add(1);
			for (int j = 1; j <= i; j++) {
				list.add(res.get(i - 1).get(j - 1) + (j == i ? 0 : res.get(i - 1).get(j)));
			}
			res.add(list);
		}
		return res;
	}

	public boolean isBalanced(TreeNode root) {
		return height(root) == -1;
	}

	private int height(TreeNode root) {
		if (root == null)
			return 0;
		int left = height(root.left);
		if (left != -1) {
			int right = height(root.right);
			if (right != -1)
				return Math.abs(left - right) > 1 ? -1 : Math.max(left, right) + 1;
		}
		return -1;
	}

	public int arrangeCoins(int n) {
		return (int) ((Math.sqrt((8.0 * n + 1) / 4) - 1));
	}

	public List<String> binaryTreePaths(TreeNode root) {
		if (root == null)
			return null;
		List<String> list = new ArrayList<>();
		String s = "";
		path(root, s, list);
		return list;
	}

	private void path(TreeNode root, String s, List<String> list) {
		if (root.left == null && root.right == null)
			list.add(s.substring(0, s.length() - 3));
		if (root.left != null)
			path(root.left, s + root.val + "->", list);
		if (root.right != null)
			path(root.right, s + root.val + "->", list);
	}

	public boolean hasCycle(ListNode head) {
		Set<ListNode> set = new HashSet<>();
		while (head != null && !set.contains(head)) {
			set.add(head);
			head = head.next;
		}
		if (head == null)
			return false;
		return true;
	}

	public List<Integer> getRow2(int rowIndex) {
		List<Integer> list = new ArrayList<>();
		list.add(1);
		for (int i = 1; i < rowIndex; i++) {
			for (int j = 1; j <= i; j++) {
				list.set(j, list.get(j - 1) + (j == i ? 0 : list.get(j)));
			}
		}
		return list;
	}

	public List<Integer> getRow(int rowIndex) {
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < rowIndex / 2 + 1; i++)
			list.add((int) zuhe(rowIndex, i));
		for (int i = rowIndex / 2 + 1; i <= rowIndex; i++)
			list.add(list.get(rowIndex - i));
		return list;
	}

	private int zuhe(int n, int m) {
		BigInteger res = new BigInteger("1");
		BigInteger div = new BigInteger("1");
		for (int i = m; i > 0; i--) {
			res = res.multiply(new BigInteger("" + n));
			n--;
			div = div.multiply(new BigInteger("" + i));
		}

		return res.divide(div).intValue();

	}

	public int guessNumber(int n) {
		int low = 1;
		while (low < n) {
			int mid = (low + n) / 2;
			if (guess(mid) == 0)
				return mid;
			if (guess(mid) == 1)
				low = mid + 1;
			else
				n = mid;
		}
		return low;
	}

	private int guess(int n) {
		if (n == 6)
			return 0;
		else if (n < 6)
			return 1;
		else
			return -1;
	}

	public List<Integer> findAnagrams(String s, String p) {
		int[] chars = new int[26];
		List<Integer> result = new ArrayList<>();

		if (s == null || p == null || s.length() < p.length())
			return result;
		for (char c : p.toCharArray())
			chars[c - 'a']++;

		int start = 0, end = 0, count = p.length();
		// Go over the string
		while (end < s.length()) {
			// If the char at start appeared in p, we increase count
			if (end - start == p.length() && chars[s.charAt(start++) - 'a']++ >= 0)
				count++;
			// If the char at end appeared in p (since it's not -1 after
			// decreasing), we decrease count
			if (--chars[s.charAt(end++) - 'a'] >= 0)
				count--;
			if (count == 0)
				result.add(start);
		}

		return result;
	}

	public boolean hasPathSum(TreeNode root, int sum) {
		if (root == null)
			return sum == 0 ? true : false;
		HashSet<Integer> set = new HashSet<>();
		addPath(root, 0, set);
		return set.contains(sum);
	}

	private void addPath(TreeNode root, int sum, HashSet<Integer> set) {
		if (root.left == null && root.right == null)
			set.add(sum + root.val);
		if (root.left != null)
			addPath(root.left, sum + root.val, set);
		if (root.right != null)
			addPath(root.right, sum + root.val, set);
	}

	public String countAndSay(int n) {
		List<Integer> res = new ArrayList<>();
		res.add(1);
		for (int i = 0; i < n; i++) {
			int tmp = res.get(0);
			int count = 0;
			List<Integer> list = new ArrayList<>();
			for (int j = 0; j < res.size(); j++) {
				if (tmp == res.get(j)) {
					count++;
					continue;
				}
				list.add(count);
				list.add(tmp);
				count = 1;
				tmp = res.get(j);
			}
			list.add(count);
			list.add(tmp);
			res = list;
		}
		String s = "";
		for (int i = 0; i < res.size(); i++) {
			s += res.get(i);
		}
		return s;
	}

	public boolean isIsomorphic(String s, String t) {
		int[] alp = new int[256];
		char[] charS = s.toCharArray();
		for (int i = 0; i < charS.length; i++) {
			if (alp[charS[i]] == 0)
				alp[charS[i]] = i + 1;
			charS[i] = (char) alp[charS[i]];
			System.out.println("" + (int) charS[i]);
		}
		char[] charT = t.toCharArray();
		alp = new int[256];
		for (int i = 0; i < charT.length; i++) {
			if (alp[charT[i]] == 0)
				alp[charT[i]] = i + 1;
			charT[i] = (char) alp[charT[i]];
			System.out.println("" + (int) charT[i]);
			if (charS[i] != charT[i])
				return false;
		}
		return true;
	}

	public boolean isValid(String s) {
		Stack<Character> stack = new Stack<>();
		String test1 = "({[";
		String test2 = ")}]";
		for (char c : s.toCharArray()) {
			if (test2.indexOf(c) != -1) {
				while (!stack.isEmpty()) {
					if (test1.indexOf(stack.peek()) != -1) {
						if (test2.indexOf(c) == test1.indexOf(stack.pop())) {
							break;
						} else
							return false;
					}
					stack.pop();
				}
			} else {
				stack.push(c);
			}

		}
		while (!stack.isEmpty()) {
			if (test1.indexOf(stack.pop()) != -1)
				return false;
		}
		return true;

	}

	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		HashSet<ListNode> set = new HashSet<>();
		while (headA != null) {
			set.add(headA);
			headA = headA.next;
		}
		while (headB != null) {
			if (!set.add(headB))
				return headB;
			headB = headB.next;
		}
		return null;
	}

	public int lengthOfLongestSubstring(String s) {
		int[] alp = new int[128];
		int max = 0;
		int cur = 0;
		for (int i = 0; i < s.length(); i++) {
			if (alp[s.charAt(i)]++ != 0) {
				cur = 1;
				alp = new int[128];
				alp[s.charAt(i)]++;
				continue;
			}
			cur++;
			max = max < cur ? cur : max;
		}
		return max;
	}
	
	public String findLongestWord(String s, List<String> d) {
    	int max = 0;
    	String res = "";
    	for (Iterator<String> it = d.iterator(); it.hasNext();){
    	    String  i = it.next();
    	    if((i.length()>max ||  (i.length()==max && i.compareTo(res)<0))&& s.contains(i)){
    	    	max = i.length();
    	    	res = i;
    	    }
    	}
    	return res;
    }
    
    
    public int countArrangement(int N) {
    	int[] nums = new int[N+1];
        return countArrangementHelper(1,N+1, nums);
    }
    
    public int countArrangementHelper(int target,int N, int[] book){
    	if(target == N) return 1;
    	int sum = 0;
    	for(int i = 1;i<book.length;i++){
    		if(book[i] == 0 && (target % i ==0 || i % target == 0)){
    			book[i] = 1;
    			sum += countArrangementHelper(target+1, N, book);
    			book[i] = 0;
    		}
    	}
    	return sum;
    }
    
    public String complexNumberMultiply(String a, String b) {
        int Rea = Integer.parseInt(a.split("\\+")[0]);
        int Ima = Integer.parseInt(a.split("\\+")[1].substring(0,a.split("\\+")[1].length()-1));
        int Reb = Integer.parseInt(b.split("\\+")[0]);
        int Imb = Integer.parseInt(b.split("\\+")[1].substring(0,b.split("\\+")[1].length()-1));
        return ""+(Rea*Reb-Ima*Imb)+"+"+(Rea*Imb+Reb*Ima)+"i";
    }
    
    public int[][] updateMatrix(int[][] matrix) {
    	Queue<Integer> qx = new LinkedList<>();
    	Queue<Integer> qy = new LinkedList<>();
    	Queue<Integer> steps = new LinkedList<>();
    	boolean[][] used = new boolean[matrix.length][matrix[0].length];
        for(int i = 0; i<matrix.length;i++){
        	for(int j=0;j<matrix[0].length;j++){
        		if(matrix[i][j] == 0) {
        			used[i][j] = true;
        			qx.add(i);
        			qy.add(j);
        			steps.add(0);
        		}
        	}
        }
        while(!qx.isEmpty()){
        	int x = qx.poll();
        	int y = qy.poll();
        	int step = steps.poll()+1;
        	if(x>0 && !used[x-1][y]){
        		matrix[x-1][y] = step;
        		used[x-1][y] =true;
        		qx.add(x-1);
        		qy.add(y);
        		steps.add(step);
        	}
        	if(y>0 && !used[x][y-1]){
        		matrix[x][y-1] = step;
        		used[x][y-1] =true;
        		qx.add(x);
        		qy.add(y-1);
        		steps.add(step);
        	}
        	if(x<matrix.length-1 && !used[x+1][y]){
        		matrix[x+1][y] = step;
        		used[x+1][y] =true;
        		qx.add(x+1);
        		qy.add(y);
        		steps.add(step);
        	}
        	if(y<matrix[0].length-1 && !used[x][y+1]){
        		matrix[x][y+1] = step;
        		used[x][y+1] =true;
        		qx.add(x);
        		qy.add(y+1);
        		steps.add(step);
        	}
        }
        return matrix;
    }
    
    public String reverseWords(String s) {
        String[] sArray = s.split(" ");
        StringBuilder res = new StringBuilder();
        for(String i : sArray){
        	res.append(new StringBuilder(i).reverse().toString()+" ");
        }
        return res.toString().trim();
    }
    
    public int findTilt(TreeNode root) {
    	if(root == null) return 0;
        return Math.abs(findTiltSum(root.left)-findTiltSum(root.right))+findTilt(root.left)+findTilt(root.right);
    }
    
    private int findTiltSum(TreeNode root){
    	if(root == null) return 0;
    	return root.val+findTiltSum(root.left) + findTiltSum(root.right);
    }
    
    
    public int arrayNesting(int[] nums) {
        boolean[] ar = new boolean[nums.length];
        int max = 0;
        int cur = 0;
        for(int i = 0; i<nums.length; i++){
        	int tmp = i;
        	while(!ar[tmp]){
        		cur++;
        		max = Math.max(max, cur);
        		ar[tmp] = true;
        		tmp = nums[tmp];
        	}
        	cur = 0;
        }
        return max;
    }
    
    public int findPaths(int m, int n, int N, int i, int j) {
    	int[][][] memo = new int[m][n][N];
    	int res = 0;
    	for(int k =1;k<=N;k++) res+=findPathsDFS(m, n, k, i, j, memo);
    	return res;
    }
    
    public int findPathsDFS(int m, int n, int N, int i, int j, int[][][] memo){
    	if(i<0||j<0||i==m||j==n) return 0;
    	if(memo[i][j][N-1] != 0) return memo[i][j][N-1];
    	if(N==1)memo[i][j][N-1] = (i==0?1:0) +(i==m-1?1:0) + (j==0?1:0) + (j==n-1?1:0);
    	else memo[i][j][N-1] = findPathsDFS(m,n,N-1,i-1,j,memo)+findPathsDFS(m,n,N-1,i,j-1,memo)+findPathsDFS(m,n,N-1,i+1,j,memo)+findPathsDFS(m,n,N-1,i,j+1,memo);
    	return memo[i][j][N-1];
    }
    
    public List<Integer> killProcess(List<Integer> pid, List<Integer> ppid, int kill) {
    	Map<Integer, List<Integer>> map = new HashMap<>();
    	for (int i = 0; i < ppid.size(); i++){
    		map.putIfAbsent(ppid.get(i), new LinkedList<>());
    		map.get(ppid.get(i)).add(pid.get(i));
    	}
    	
        Queue<Integer> queue = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        queue.add(kill);
        while(!queue.isEmpty()){
        	int cur = queue.poll();
            res.add(cur);
            List<Integer> tmp = map.get(cur);
            if (tmp == null) continue;
            queue.addAll(tmp);
        }
        return res;
    }
    
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null) return t2;
        if(t2 == null) return t1;
        TreeNode res = new TreeNode(t1.val+t2.val);
        res.left = mergeTrees(t1.left, t2.left);
        res.right = mergeTrees(t1.right, t2.right);
        return res;
    }
	
    public ListNode oddEvenList(ListNode head) {
        ListNode odd = head;
        if(head == null || head.next == null) return head;
        ListNode evenHead = head.next;
        ListNode evenTail = evenHead;
        while(evenTail.next != null && evenTail.next.next != null){
        	odd.next = evenTail.next;
        	evenTail.next = evenTail.next.next;
        	odd= odd.next;
        	evenTail = evenTail.next;
        }
        if(evenTail.next != null) {
        	odd.next = evenTail.next;
        	odd = odd.next;
        }
        odd.next = evenHead;
        return head;
    }
    
    public int findTargetSumWays(int[] nums, int S) {
    	Map<Integer, Integer> map = new HashMap<>();
    	//map.put(0, 1);
        for(int n:nums){
        	Map<Integer, Integer> mapAdd = new HashMap<>();
        	Map<Integer, Integer> mapSub = new HashMap<>();
        	for(Integer i : map.keySet()){
        		mapAdd.put(i+n, map.get(i));
        		mapSub.put(i-n, map.get(i));
        	}
        	for(Integer i: mapAdd.keySet()){
        		mapSub.put(i, mapAdd.get(i)+mapSub.getOrDefault(i, 0));
        	}
        	map = mapSub;
        }
        return map.getOrDefault(S, 0);
    }
    
    public int[] nextGreaterElements(int[] nums) {
    	if(nums.length == 0) return nums;
    	Stack<Integer> index = new Stack<>();
    	int res[] = new int[nums.length];
    	for (int i = 0; i < nums.length; i++) {
    		res[i] = -1;
			while(!index.isEmpty() && nums[index.peek()]<nums[i]) res[index.pop()] = nums[i];
			index.push(i);
		}
    	for (int n : nums)  while(!index.isEmpty() && nums[index.peek()]<n) res[index.pop()] = n;
        return res;
    }
    
    public boolean checkPerfectNumber(int num) {
    	if(num == 1 )return false;
    	int sum = 1;
    	for(int i = 2, j = num/2; i <j ; i++){
    		if(num%i==0) {
    			j=num/i;
    			sum+=i+j;
    		}
    	}
    	return sum == num;
    }
    
    public String optimalDivision(int[] nums) {
      if(nums.length == 1) return "" + nums[0];
      if(nums.length == 2) return "" + nums[0] + "/" +nums[1];
      String res = ""+nums[0]+"/(" +nums[1];
      for(int i = 2;i<nums.length;i++) res+="/"+nums[i];
      return res + ")";
    }
    
    public int subarraySum(int[] nums, int k) {
    	if(nums.length == 0) return 0;
        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0;
        for(int n:nums){
        	sum+=n;
        	map.put(sum, map.getOrDefault(sum, 0)+1);
        }
        sum = 0;
        int res = map.getOrDefault(k, 0);
        for(int n:nums){
        	sum+=n;
        	if(map.get(sum) > 1) map.put(sum, map.get(sum)-1);
        	else map.remove(sum);
        	res+=map.getOrDefault(k+sum, 0);
        }
        return res;
    }
    
    public String mirroring(String s) {
        String x = s.substring(0, (s.length()) / 2);
        return x + (s.length() % 2 == 1 ? s.charAt(s.length() / 2) : "") + new StringBuilder(x).reverse().toString();
    }
    
    public String nearestPalindromic(String n) {
        int mid = n.length()/2;
        Long num = Long.parseLong(n);
        String first= nearestPalindromicHelper(n, mid);
        int plus = (int) Math.pow(10, mid);
        Long sub = num - Long.parseLong(first);
        if(sub != 0 &&( Math.abs(sub) <plus/2 || sub == plus)) return first;
        if(sub<=0) plus = -plus;
        num+=plus;
        return nearestPalindromicHelper(Long.toString(num), mid);
    }
    
    private String nearestPalindromicHelper(String n, int mid) {
    	String s=n.substring(0, mid);
        if(n.length()%2 == 1) s += n.charAt(mid);
        for(int i = mid-1; i>=0;i--) s+=n.charAt(i);
        return s;
	}
    
    
    public boolean checkInclusion(String s1, String s2) {
        int[] chars = new int[26];
        if (s1 == null || s2 == null || s1.length() > s2.length())
            return false;
        for (char c : s1.toCharArray())
            chars[c-'a']++;
    
        int start = 0, end = 0, count = s1.length();
        // Go over the string
        while (end < s2.length()) {
            // If the char at start appeared in p, we increase count
            if (end - start == s1.length() && chars[s2.charAt(start++)-'a']++ >= 0)
                count++;
            // If the char at end appeared in p (since it's not -1 after decreasing), we decrease count
            if (--chars[s2.charAt(end++)-'a'] >= 0)
                count--;
            if (count == 0)
                return true;
        }
        
        return false;
    }
    
    public int minDistance(String word1, String word2) {
    	int[] dp = new int[word2.length()+1];
        for (int i = 1; i<=word1.length();i++){
        	int prev = 0;
        	for(int j = 1;j<dp.length;j++){
        		int cur = dp[j];
        		if(word1.charAt(i-1) == word2.charAt(j-1)) dp[j] = prev+1;
        		else dp[j] = Math.max(cur, dp[j-1]);
        		prev = cur;
        	}
        }
        return word1.length()+word2.length()-2*dp[word2.length()];
    }
    
    public int findLHS(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int n:nums) map.put(n, map.getOrDefault(n, 0)+1);
        int max = 0;
        for(int n:map.keySet()){
        	if(!map.containsKey(n+1)) continue;
        	max = Math.max(max, map.get(n)+map.get(n+1));
        }
        return max;
    }
    
    public int maxCount(int m, int n, int[][] ops) {
    	int x = m;
    	int y = n;
        for(int[]l:ops){
        	x = Math.min(x, l[0]);
        	y = Math.min(y, l[1]);
        }
        return x*y;
    }
    
    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        int dis12 = validSquaredis(p1,p2);
        int dis13 = validSquaredis(p1, p3);
        int dis14 = validSquaredis(p1, p4);
        if(dis12 == 0 || dis13 == 0 ||dis14==0) return false;
        if(dis12 == dis14) validSquareswap(p3, p4);
        else if(dis13 == dis14) validSquareswap(p2, p4);
        else if(dis12 == dis13) ;
        else return false;
        if((p3[1]-p1[1])*(p2[1]-p1[1]) != (p1[0]-p3[0])*(p2[0]-p1[0])) return false; 
        if(validSquaredis(p2, p4) != validSquaredis(p3, p4)) return false;
        if((p3[1]-p4[1])*(p2[1]-p4[1]) != (p4[0]-p3[0])*(p2[0]-p4[0])) return false;
        return true;
    }
    
    private void validSquareswap(int[]p1, int[]p2){
    	int[]tmp = new int[2];
    	tmp[0] = p1[0];
    	tmp[1] = p1[1];
    	p1[0] = p2[0];
    	p1[1] = p2[1];
    	p2[0] = tmp[0];
    	p2[1] = tmp[1];
    }
    
    private int validSquaredis(int[]p1, int[]p2) {
		return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]);
	}
    
    public String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map1 = new HashMap<>();
        Map<String,Integer> res = new HashMap<>();
        if(list2.length < list1.length) {
        	String[] tmp = list2;
        	list2=list1;
        	list1 = tmp;
        }
        for(int i = 0;i<list1.length;i++) map1.put(list1[i], i);
        int min = Integer.MAX_VALUE;
        for(int i = 0;i<list2.length &&i<=min;i++){
        	if(!map1.containsKey(list2[i])) continue;
        	if( min< i+map1.get(list2[i])) continue;
        	min =  i+map1.get(list2[i]);
        	res.put(list2[i],min);
        }
        List<String> list= new ArrayList<>();
        for(String s : res.keySet()) if(res.get(s) == min) list.add(s);
        String[] ss = new String[list.size()];
        for(int i = 0;i<ss.length;i++) ss[i] = list.get(i);
        return ss;
    }
    
    public int findIntegers(int num) {
    	if(num<2) return num+1;
        int[] dp = new int[num+1];
        dp[0] = 1;
        dp[1] = 2;
        int bit = 2;
        for(int i = 2;i<dp.length;i++){
        	if(i==bit<<1) bit=bit<<1;
        	if(i-bit>=bit>>>1) dp[i] =dp[i-1];
        	else dp[i] = dp[i-1]+(i-bit==0?1:(dp[i-bit]==dp[i-bit-1]?0:1));
        }
        return dp[num];
    }

    public boolean canPlaceFlowers(int[] flowerbed, int n) {
    	if(n== 0) return true;
    	boolean prevprev = true;
    	boolean prev = flowerbed[0] == 0?true:false;
    	if(flowerbed.length == 1) return prev;
    	int count = 0;
        for(int i = 1; i<flowerbed.length;i++){
        	boolean cur = flowerbed[i] == 0?true:false;
        	if(prevprev && prev && cur){
        		count ++;
        		if(count >= n) return true;
        		prev = false;
        	}
        	prevprev = prev;
        	prev = cur;
        }
        return n-count == 1?prev&&prevprev:false;
    }
    
    public int triangleNumber(int[] nums) {
        Arrays.sort(nums);
        int count = 0;
        for(int i = 0; i<nums.length-2;i++){
        	for(int j = i+1;j<nums.length-1;j++){
        		int k = i+2;
        		for(;k<nums.length && nums[i] + nums[j] > nums[k];k++){        		
        		}
        		count += k-j-1;
        	}
        }
        return count;
    }
    
    public int minSubArrayLen(int s, int[] nums) {
    	if(nums.length == 0) return 0;
        int start = 0;
        int end = 0;
        int sum = 0;
        int min = Integer.MAX_VALUE;
        while(end<nums.length){
        	if(sum >=s) {
        		min = Math.min(min, end-start);
        		sum-=nums[start++];
        	}
        	else if(sum <s) sum+=nums[end++];
        }
        while(sum >=s) {
    		min = Math.min(min, end-start);
    		sum-=nums[start++];
    	}
        return min==Integer.MAX_VALUE?0:min;
    }
    
    public String tree2str(TreeNode t) {
    	if(t == null) return "";
    	String left = tree2str(t.left);
    	String right = tree2str(t.right);
    	if(left == "" && right =="") return ""+t.val;
    	if(right == "" ) return ""+t.val+"("+left+")";
        return ""+t.val+"("+left+")("+right+")";
    }
    
    public int trap(int[] height) {
        int count = 0;
        Stack<Integer> stack = new Stack<>();
        boolean start =false;
        int max = 0;
    	for(int i = 0; i<height.length;i++){
    		if(!start) {
    			if(height[i]!=0){
        			stack.push(height[i]);
        			max = height[i];
        			start = true;
    			}
    			continue;
    		}
    		if(height[i]-stack.peek() <=0){
    			stack.push(height[i]);
    		}
    		else if(height[i] -stack.peek() > 0 && height[i] < max){
    			int tmp = 0;
    			while(height[i] - stack.peek() > 0){
    				count+=height[i]-stack.pop();
    				tmp++;
    			}
    			for(int j = 0; j<tmp+1;j++) stack.push(height[i]);
    		}
    		else {
        		while(!stack.isEmpty()){
        			count+=max-stack.pop();
        		}
        		stack.push(height[i]);
        		max = height[i];
    		}
    	}
    	return count;
    }
    
    public int leastInterval(char[] tasks, int n) {
    	int[] aph = new int[26];
        for(char c : tasks) aph[c-'A']++;
        PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator <Integer> () {
	@Override
	public int compare(Integer a, Integer b) {
		return aph[b]-aph[a];
	}
	  });
        
        for(int i =0;i<aph.length;i++) if(aph[i] != 0 ) pq.add(i);
        int count = 0;
        Queue<Integer> wait = new LinkedList<>();
        for(int i = 0; i<n;i++)wait.add(-1);
        while(!pq.isEmpty() || (!wait.isEmpty() && leastIntervalHelper(wait))){
        	if(!pq.isEmpty()) {
        		int tmp = pq.poll();
        		if(--aph[tmp]!=0) 
        			wait.add(tmp);
        		else wait.add(-1);
        	}
        	else wait.add(-1);
        	count++;
        	if(!wait.isEmpty()){
        		if(wait.peek() != -1) pq.add(wait.poll());
        		else wait.poll();
        	}
        }
        return count;
    }
    
    private Boolean leastIntervalHelper( Queue<Integer> wait){
    	for (Integer i : wait) {
			if (i !=-1) return true;
		}
    	return false;
    }
    
    public int smallestFactorization(int a) {
    	int last = a;
    	while(last/10!=0) last=a/10;
        for(int i = last+1; i<=9;i++ ){
        	
        }
        return last;
    }
    
    public int maxDistance(List<List<Integer>> arrays) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for(int i = 0; i<arrays.size();i++){
        	List<Integer> list = arrays.get(i);
        	if(map.containsKey(list.get(0))) map.put(list.get(0), -1);
        	else map.put(list.get(0), i);
        	if(map.containsKey(list.get(list.size()-1))) map.put(list.get(list.size()-1), -1);
        	else map.put(list.get(list.size()-1), i);
        }
        if( map.get(map.lastKey()) != map.get(map.firstKey()) || map.get(map.lastKey()) == -1 || map.get(map.firstKey()) == -1) return map.lastKey()-map.firstKey();
        else {
        	int lastkey = map.lastKey();
        	map.remove(lastkey);
        	int a = map.lastKey() - map.firstKey();
        	map.remove(map.firstKey());
        	int b = lastkey - map.firstKey();
        	return a>b?a:b;
        }
    }
    
    public TreeNode addOneRow(TreeNode root, int v, int d) {
    	if(root == null) return null;
        if(d==1) {
        	TreeNode res = new TreeNode(v);
        	res.left = root;
        	return res;
        }
        if(d==2){
        	TreeNode left = new TreeNode(v);
        	TreeNode right = new TreeNode(v);
        	left.left = root.left;
        	right.right = root.right;
        	root.left = left;
        	root.right = right;	
        }
        else {
        	addOneRow(root.left, v, d-1) ;
        	addOneRow(root.right, v, d-1);
        }
        return root;
    }
    
    public int findPeakElement(int[] nums) {
    	for(int i  = 1; i <nums.length; i++){
    		if(nums[i] < nums[i-1]) return i-1;
    	}
    	return nums.length-1;
    }
    
    public String shortestPalindrome(String s) {
        Stack<Character> stack = new Stack<>();
        boolean reset = true;
        int i = s.length()/2+1;
        for(;i>0;i--){
        	if(reset){
        		stack.clear();
        		for(int j = 0; j<i;j++) stack.push(s.charAt(j));
        	}
        	if(s.charAt(i+1) == stack.pop()) {
        		for(int k = i; k<s.length() || stack.isEmpty() ; k++){
        			if(s.charAt(k) != stack.pop()) {
        				reset = true;
        				break;
        			}
        		}
        	}
        	
        }
        String res = "";
        for(int k = i+1;i<s.length();i++) res = s.charAt(k) + res;
        return res+s;
        
    }

	public static void main(String[] args) {
		test a = new test();
		System.out.println(""+a.leastInterval(new char[]{'A','A','A','E','D','C'}, 2));
	}
}