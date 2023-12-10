using System.Text;

namespace BitTensor.Core;

public partial class Tensor
{
    public override string ToString() => $"Tensor #{Id}, shape={Shape.Serialize()}";

    public string ToDataString(int dimsPerLine = 1)
    {
        if (IsEmpty)
        {
            return "[]";
        }

        if (IsScalar)
        {
            return Values[0].ToString("0.00#");
        }

        var sb = new StringBuilder();
        var products = new List<int>();
        var product = 1;
        for (var i = Dimensions - 1; i >= 0; --i)
        {
            product *= Shape[i];
            products.Add(product);
        }

        for (var i = 0; i < Values.Length; ++i)
        {
            var opens = products.Count(p => (i) % p == 0);
            var closes = products.Count(p => (i + 1) % p == 0);
            var value = Values[i].ToString("0.00#").PadLeft(Dimensions > 1 ? 6 : 0);

            if (opens > 0)
                sb.Append(new string(' ', Dimensions - opens));

            sb.Append(new string('[', opens));

            if (opens > 0)
                sb.Append(" ");

            sb.Append($"{value} ");
            sb.Append(new string(']', closes));

            if (closes >= dimsPerLine && i != Values.Length - 1)
                sb.AppendLine();
        }

        return sb.ToString();
    }
}