// Self-check for the core logic: build the ufc_test target and run it.
#undef NDEBUG
#include "ufc.hpp"
#include <cassert>

using namespace ufc;

int main() {
    auto r = split_csv(R"(a,"Las Vegas, Nevada","say ""hi""",)");
    assert(r.size() == 4 && r[1] == "Las Vegas, Nevada" && r[2] == "say \"hi\"" && r[3].empty());

    assert(days_from_civil(1970, 1, 1) == 0);
    assert(parse_date("March 23, 2024") == days_from_civil(2024, 3, 23));
    assert(date_str(parse_date("January 05, 1999")) == "1999-01-05");
    assert(date_str(days_from_civil(2000, 2, 29)) == "2000-02-29");
    assert(method_of("TKO - Doctor's Stoppage") == KO && method_of("Decision - Split") == DEC);

    Record a, b;
    elo_update(a, b);
    assert(a.elo > 1500 && std::abs(a.elo + b.elo - 3000) < 1e-9);  // zero-sum

    // Shrinkage: a debutant gets the league prior; a long career converges to its raw rate.
    Stats L{};
    L.slpm = 4;
    assert(std::abs(stats(Record{}, L).slpm - 4) < 1e-9);
    Record vet;
    vet.secs = 600 * 60, vet.sig = 6 * 600;
    assert(std::abs(stats(vet, L).slpm - 6) < 0.05);

    // Logistic regression learns the sign of a noisy relation and stays antisymmetric.
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    for (int i = -50; i <= 50; ++i) X.push_back({double(i), 0.0}), y.push_back((i > 0) != (i % 7 == 0));
    Logit m;
    m.fit(X, y);
    assert(m.prob({10, 0}) > 0.6 && std::abs(m.prob({10, 0}) + m.prob({-10, 0}) - 1) < 1e-12);

    std::puts("all tests passed");
}
