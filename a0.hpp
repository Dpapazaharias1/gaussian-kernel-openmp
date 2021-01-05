/*  Demetrios
 *  Papazaharias
 *  dvpapaza
 */

#ifndef A0_HPP
#define A0_HPP

#include <vector>

float kernel(float x)
{
    return std::exp(-std::pow(x, 2) / 2);
}

void gaussian_kde(int n, float h, const std::vector<float> &x, std::vector<float> &y)
{

    float scale = std::sqrt(2 * std::acos(-1)) * n * h;

#pragma omp parallel default(none) share(y, scale, n, h) firstprivate(x)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                y[i] += kernel((x[i] - x[j]) / h);
            }
            y[i] /= scale;
        }
    }
} // gaussian_kde

#endif // A0_HPP
