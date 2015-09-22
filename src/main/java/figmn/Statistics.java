/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package figmn;

/**
 *
 * @author rafa
 */
public class Statistics {
    private static double gammaln(double xx)
    {
        double y = xx;
        double x = xx;
        double tmp = x + 5.2421875;
        tmp = (x+0.5)*Math.log(tmp)-tmp;
        double ser = 0.999999999999997092;
        for (int i = 0; i < 14; i++)
        {
            ser += COF[i]/++y;
        }
        return tmp + Math.log(2.5066282746310005*ser/x);
    }

    private static final double[] COF = {
        57.1562356658629235,
        -59.5979603554754912,
        14.1360979747417471,
        -0.491913816097620199,
        0.339946499848118887e-4,
        0.465236289270485756e-4,
        -0.983744753048795646e-4,
        0.158088703224912494e-3,
        -0.210264441724104883e-3,
        0.217439618115212643e-3,
        -0.164318106536763890e-3,
        0.844182239838527433e-4,
        -0.261908384015714087e-4,
        0.368991826595316234e-5
    };
    
    /**
* Returns the inverse chi-squared distribution. Uses the method given in
* Best and Roberts 1975. Makes calls to private functions using the methods
* of Bhattacharjee 1970 and Odeh and Evans 1974. All converted to Java by
* the author (yes, the author knows FORTRAN!)
* @param p The p-value
* @param v The number of degrees of freedom
* @return The percentage point
*/
    public static double chi2inv(double p, double v) {
        if (p < 0.000002)
        {
        return 0.0;
        }
        if (p > 0.999998)
        {
        p = 0.999998;
        }

        double xx = 0.5 * v;
        double c = xx - 1.0;
        double aa = Math.log(2);
        double g = gammaln(v/2.0);
        double ch;
        if (v > (-1.24 * Math.log(p)))
        {
        if (v > 0.32)
        {
        //3
        double x = gauinv(p);
        double p1 = 0.222222 / v;
        ch = v * Math.pow(x * Math.sqrt(p1) + 1.0 - p1, 3);
        if (ch > (2.2 * v + 6.0))
        {
        ch = -2.0 * (Math.log(1.0 - p) - c * Math.log(0.5 * ch) + g);
        }
        }
        else
        {
        //1+2
        ch = 0.4;
        double q;
        double a = Math.log(1.0-p);
        do
        {
        q = ch;
        double p1 = 1.0 + ch * (4.67 + ch);
        double p2 = ch * (6.73 + ch * (6.66 + ch));
        double t = -0.5 + (4.67 + 2.0 * ch) / p1 -
        (6.73 + ch * (13.32 + 3.0 * ch)) / p2;
        ch = ch - (1.0 - Math.exp(a + g + 0.5*ch + c*aa) * p2 / p1) / t;
        }
        while (Math.abs(q/ch - 1.0) >= 0.01);
        }
        }
        else
        {
        //START
        ch = Math.pow(p * xx * Math.exp(g + xx * aa),1.0 / xx);
        }
        double q;
        do
        {
        //4 + 5
        q = ch;
        double p1 = 0.5 * ch;
        double p2 = p - gammaintegral(xx,p1);
        double t = p2 * Math.exp(xx * aa + g + p1 - c * Math.log(ch));
        double b = t / ch;
        double a = 0.5 * t - b * c;
        double s1 = (210.0 + a * (140.0 + a * (105.0 + a * (84.0 + a * (70.0 + 60.0 * a))))) / 420.0;
        double s2 = (420.0 + a * (735.0 + a * (966.0 + a * (1141.0 + 1278.0 * a)))) / 2520.0;
        double s3 = (210.0 + a * (462.0 + a * (707.0 + 932.0 * a))) / 2520.0;
        double s4 = (252.0 + a * (672.0 + 1182.0 * a) + c * (294.0 + a * (889.0 + 1740.0 * a))) / 5040.0;
        double s5 = (84.0 + 264.0 * a + c * (175.0 + 606.0 * a)) / 2520.0;
        double s6 = (120.0 + c * (346.0 + 127.0 *c)) / 5040.0;
        ch = ch + t * (1.0+0.5*t*s1-b*c*(s1-b*(s2-b*(s3-b*(s4-b*(s5-b*s6))))));
        }
        while (Math.abs(q / ch - 1.0) > E);
        return ch;
    }
    
    private static double gammaintegral(double p, double x)
    {
	double g = gammaln(p);
	double factor = Math.exp(p*Math.log(x) - x - g);
	double gin;
	if ((x > 1.0) && (x > p))
	{
	    boolean end = false;
	    double a = 1.0 - p;
	    double b = a+x+1.0;
	    double term = 0.0;
	    double[] pn = new double[6];
	    pn[0] = 1.0;
	    pn[1] = x;
	    pn[2] = x+1.0;
	    pn[3] = x*b;
	    gin = pn[2] / pn[3];
	    do
	    {
		double rn;
		a++;
		b = b + 2.0;
		term++;
		double an = a * term;
		for (int i = 0; i <= 1; i++)
		{
		    pn[i+4] = b * pn[i+2]-an*pn[i];
		}
		if (pn[5] != 0.0)
		{
		    rn = pn[4] / pn[5];
		    double diff = Math.abs(gin - rn);
		    if (diff < E*rn)
		    {
			end = true;
		    }
		    else
		    {
			gin = rn;
		    }
		}
		if (!end)
		{
		    for (int i = 0; i < 4; i++)
		    {
			pn[i] = pn[i+2];
		    }
		    if (Math.abs(pn[5]) >= OFLO)
		    {
			for (int i = 0; i < 4; i++)
			{
			    pn[i] = pn[i] / OFLO;
			}
		    }
		}		
	    }
	    while (!end);
	    gin = 1.0 - factor*gin;
	}
	else
	{
	    gin = 1.0;
	    double term = 1.0;
	    double rn = p;
	    do
	    {
		rn++;
		term = term * x / rn;
		gin = gin + term;
	    }
	    while (term > E);
	    gin = gin * factor / p;
	}
	return gin;
    }
    
    private static double gauinv(double p)
    {
        if (p == 0.5)
        {
        return 0.0;
        }
        double ps = p;
        if (ps > 0.5)
        {
        ps = 1 - ps;
        }
        double yi = Math.sqrt(Math.log(1.0 / (ps * ps)));
        double gauinv = yi + ((((yi * p4 + p3) * yi + p2) * yi + p1) * yi + p0) /
        ((((yi * q4 + q3) * yi + q2) * yi + q1) * yi + q0);
        if (p < 0.5)
        {
        return -gauinv;
        }
        else
        {
        return gauinv;
        }
    }
    
    private static final double p0 = -0.322232431088;
    private static final double p1 = -1.0;
    private static final double p2 = -0.342242088547;
    private static final double p3 = -0.204231210245e-1;
    private static final double p4 = -0.453642210148e-4;
    private static final double q0 = 0.993484626060e-1;
    private static final double q1 = 0.588581570495;
    private static final double q2 = 0.531103462366;
    private static final double q3 = 0.103537752850;
    private static final double q4 = 0.38560700634e-2;
    private static final double E = 10e-6;
    private static final double OFLO = 10e30;
    
}
