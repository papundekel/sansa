#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <vector>

namespace dpl = oneapi::dpl;

extern "C" std::int64_t core_icf(std::int64_t n,
                                 const float* Av,
                                 const std::int64_t* Ar,
                                 const std::int64_t* Ap,
                                 float* Lv,
                                 std::int64_t* Lr,
                                 std::int64_t* Lp,
                                 std::int64_t max_nnz,
                                 float shift)
{
    auto nnz = std::int64_t(0);
    auto s = std::vector<std::int64_t>(n, 0);
    auto t = std::vector<std::int64_t>(n, 0);
    auto l = std::vector<std::int64_t>(n, -1);
    auto a = std::vector<float>(n, 0);
    auto r = std::vector<float>(n, 0);
    auto b = std::vector<bool>(n, false);
    auto c = std::vector<std::int64_t>(n);
    auto d = std::vector<float>(n, shift);

    auto aa_buffer = std::vector<float>(n);
    auto largest_indices_buffer = std::vector<std::int64_t>(max_nnz);
    auto sorted_cc_buffer = std::vector<std::int64_t>(n);

    for (const auto j : std::views::iota(0, n))
    {
        for (const auto idx : std::views::iota(Ap[j], Ap[j + 1]))
        {
            const auto i = Ar[idx];

            if (i == j)
            {
                d[j] += Av[idx];
                t[j] = idx + 1;
            }

            if (i >= j)
            {
                r[j] += std::abs(Av[idx]);
            }
        }
    }

    for (const auto j : std::views::iota(0, n))
    {
        auto c_n = std::int64_t(0);

        for (const auto idx : std::views::iota(t[j], Ap[j + 1]))
        {
            const auto i = Ar[idx];
            const auto L_ij = Av[idx];

            if (L_ij != 0 && i > j)
            {
                a[i] += L_ij;

                if (!b[i])
                {
                    b[i] = true;
                    c[c_n] = i;
                    ++c_n;
                }
            }
        }

        for (auto k = l[j]; k != -1;)
        {
            const auto k0 = s[k] + 1;
            const auto k1 = Lp[k + 1];
            const auto k2 = l[k];
            const auto L_jk = Lv[k0 - 1];

            if (k0 < k1)
            {
                s[k] = k0;
                auto i = Lr[k0];
                l[k] = l[i];
                l[i] = k;

                for (const auto idx : std::views::iota(k0, k1))
                {
                    i = Lr[idx];
                    const auto L_ik = Lv[idx];
                    a[i] -= L_ik * L_jk;
                    if (!b[i])
                    {
                        b[i] = true;
                        c[c_n] = i;
                        ++c_n;
                    }
                }
            }

            k = k2;
        }

        if (d[j] < 0)
        {
            return -1;
        }

        const auto max_j_nnz = (max_nnz - nnz) / (n - j);

        const auto cc_view = c | std::views::take(c_n);

        if (c_n > max_j_nnz)
        {
            const auto aa_view = cc_view | std::views::transform(
                                               [&a](std::int64_t i)
                                               {
                                                   return std::abs(a[i]);
                                               });

            std::copy(dpl::execution::unseq,
                      aa_view.begin(),
                      aa_view.end(),
                      aa_buffer.begin());

            auto largest_indices = std::span(largest_indices_buffer.begin(), max_j_nnz);

            const auto aa_indices_begin = dpl::counting_iterator<std::size_t>(0uz);

            std::partial_sort_copy(dpl::execution::unseq,
                                   aa_indices_begin,
                                   aa_indices_begin + aa_view.size(),
                                   largest_indices.begin(),
                                   largest_indices.end(),
                                   [&aa_buffer](std::size_t a, std::size_t b)
                                   {
                                       return aa_buffer[a] > aa_buffer[b];
                                   });

            const auto b_dereference_view = std::views::transform(
                [&b](std::int64_t i)
                {
                    return b[i];
                });

            std::ranges::fill(cc_view | b_dereference_view, false);

            std::ranges::fill(largest_indices |
                                  std::views::transform(
                                      [cc_view](std::int64_t i)
                                      {
                                          return cc_view[i];
                                      }) |
                                  b_dereference_view,
                              true);
        }

        d[j] = std::sqrt(d[j]);
        Lv[nnz] = d[j];
        Lr[nnz] = j;
        nnz += 1;
        s[j] = nnz;

        const auto sorted_cc = std::span(sorted_cc_buffer.begin(), cc_view.size());

        std::copy(dpl::execution::unseq,
                  cc_view.begin(),
                  cc_view.end(),
                  sorted_cc.begin());
        std::sort(sorted_cc.begin(), sorted_cc.end());

        for (const auto i : sorted_cc)
        {
            const auto L_ij = a[i] / d[j];
            d[i] -= L_ij * L_ij;

            if (b[i])
            {
                Lv[nnz] = L_ij;
                Lr[nnz] = i;
                nnz += 1;
            }

            a[i] = 0;
            b[i] = false;
        }

        Lp[j + 1] = nnz;

        if (Lp[j] + 1 < Lp[j + 1])
        {
            const auto i = Lr[Lp[j] + 1];
            l[j] = l[i];
            l[i] = j;
        }
    }

    return nnz;
}
