import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class HelloWorld {
    
    public static void main(String[] args) {
		// ����Calendar����
		Calendar c = Calendar.getInstance();
        
		// ��Calendar����ת��ΪDate����
		Date date = c.getTime();
        
		// ����SimpleDateFormat����ָ��Ŀ���ʽ
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        
		// ������ת��Ϊָ����ʽ���ַ���
		String now = sdf.format(date);
		System.out.println("��ǰʱ�䣺" + now);
		
		int j = 0xffffffff;
		System.out.println(j);
		

    }
}