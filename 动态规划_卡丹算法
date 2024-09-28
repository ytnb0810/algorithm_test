import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StreamTokenizer;

public class _2 {
    public static void main(String[] args) throws IOException {
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));
        in.nextToken();
        int n = (int) in.nval;
        int s[] = new int[100005];
        for (int i = 0; i < n; i++) {
            in.nextToken();
            s[i] = (int) in.nval;
        }
        int max = 0, sum = 0;
        for (int i = 0; i < n; i++) {
            sum += s[i];
            if (sum > max) {
                max = sum;
            }
            if (sum <= 0) {
                sum = 0;
            }
        }
        out.println(max);
        out.flush();
    }
}
