// UFC analytics CLI.
//   ufc evaluate                        backtest the winner model and metric projections on the latest 20% of fights
//   ufc profile "<fighter>"             career metrics, style archetype, movement/habit profile
//   ufc predict "<A>" "<B>" [3|5]       win/method probabilities and projected fight metrics
#include "ufc.hpp"
#include <cctype>
#include <numeric>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>  // SetConsoleOutputCP, so accented names print correctly
#endif

using namespace ufc;

static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static std::string data_dir() {
    for (const char* d : {"data", "../data", "../../data"})
        if (std::ifstream(std::string(d) + "/large_dataset.csv")) return d;
    throw std::runtime_error("data/large_dataset.csv not found; run from the ufc_analytics folder");
}

// Exact (case-insensitive) name, else a unique substring match; prints candidates otherwise.
static const Record* lookup(const World& w, const std::string& query) {
    std::string q = lower(query);
    std::vector<const Record*> hits;
    for (auto& [name, r] : w.rec) {
        std::string l = lower(name);
        if (l == q) return &r;
        if (l.find(q) != std::string::npos) hits.push_back(&r);
    }
    if (hits.size() == 1) return hits[0];
    if (hits.empty()) {
        std::printf("No fighter matching \"%s\".\n", query.c_str());
        return nullptr;
    }
    std::sort(hits.begin(), hits.end(), [](auto a, auto b) { return a->n > b->n; });
    std::printf("\"%s\" matches several fighters:\n", query.c_str());
    for (size_t i = 0; i < hits.size() && i < 15; ++i) std::printf("  %s (%d UFC fights)\n", hits[i]->name.c_str(), hits[i]->n);
    return nullptr;
}

static Logit train(const World& w, size_t end, bool corner) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    for (size_t i = 0; i < end; ++i) {
        X.push_back(w.samples[i].x);
        if (!corner) X.back().back() = 0;
        y.push_back(w.samples[i].y);
    }
    Logit m;
    m.fit(X, y);
    return m;
}

// ---------- evaluate ----------

struct Score {
    double hit = 0, loss = 0;
    int n = 0;
    void add(double p, int y) { hit += (p > 0.5) == (y == 1), loss -= std::log(y ? p : 1 - p), ++n; }
    void print(const char* label) const { std::printf("  %-44s %6.1f%%   %.3f   (n=%d)\n", label, 100 * hit / n, loss / n, n); }
};

struct Err {
    double model = 0, own = 0, league = 0, wsum = 0;
    void add(double m, double o, double l, double actual, double wt) {
        model += wt * std::abs(m - actual), own += wt * std::abs(o - actual), league += wt * std::abs(l - actual), wsum += wt;
    }
    void print(const char* label) const { std::printf("  %-26s %10.3f %12.3f %12.3f\n", label, model / wsum, own / wsum, league / wsum); }
};

static int evaluate(const World& w) {
    size_t n = w.samples.size(), cut = n * 4 / 5;
    std::printf("%zu fights, %zu fighters, %s to %s\n", n, w.rec.size(), date_str(w.fights.front().date).c_str(),
                date_str(w.ref_date).c_str());
    std::printf("Train on the first %zu fights, test on the last %zu (%s onward). Features use pre-fight history only.\n\n",
                cut, n - cut, date_str(w.fights[cut].date).c_str());

    Logit plain = train(w, cut, false), cornered = train(w, cut, true);
    double red_rate = 0;
    for (size_t i = 0; i < cut; ++i) red_rate += w.samples[i].y;
    red_rate /= cut;

    Score red, elo, model, model_c, vets;
    Err slpm, acc, td, ctrl;
    for (size_t i = cut; i < n; ++i) {
        const Sample& s = w.samples[i];
        auto x0 = s.x;
        x0.back() = 0;
        double p = plain.prob(x0);
        red.add(red_rate, s.y);
        elo.add(1 / (1 + std::pow(10, -s.x[0] / 4)), s.y);  // x[0] is the Elo gap / 100
        model.add(p, s.y);
        model_c.add(cornered.prob(s.x), s.y);
        if (s.n_red >= 3 && s.n_blue >= 3) vets.add(p, s.y);

        const Fight& f = w.fights[s.fight];
        double min = f.secs / 60;
        for (int side = 0; side < 2; ++side) {
            const Stats &a = side ? s.blue : s.red, &b = side ? s.red : s.blue;
            const Corner& c = f.c[side];
            const Stats& L = w.L;
            slpm.add(project(a.slpm, b.sapm, L.slpm), a.slpm, L.slpm, c.sig / min, min);
            if (c.sig_att > 0)
                acc.add(project(a.sig_acc, 1 - b.str_def, L.sig_acc), a.sig_acc, L.sig_acc, c.sig / c.sig_att, c.sig_att);
            td.add(project(a.td15, b.td_abs15, L.td15), a.td15, L.td15, 15 * c.td / min, min);
            ctrl.add(project(a.ctrl, b.ctrl_abs, L.ctrl), a.ctrl, L.ctrl, c.ctrl / f.secs, min);
        }
    }
    std::printf("Winner prediction on test fights               accuracy  log loss\n");
    red.print("Always pick the red corner (baseline)");
    elo.print("Elo rating only");
    model.print("Logistic model, corner-blind");
    model_c.print("Logistic model + corner");
    vets.print("Corner-blind, both fighters 3+ UFC fights");

    std::printf("\nStrongest features of the corner-blind model (standardized; + favours having more of it)\n");
    std::vector<size_t> order(FEATURES.size() - 1);  // red_corner is zeroed in this model
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return std::abs(plain.w[a]) > std::abs(plain.w[b]); });
    for (size_t i = 0; i < 8; ++i) std::printf("  %-14s %+.3f\n", FEATURES[order[i]].c_str(), plain.w[order[i]]);

    std::printf("\nFight-metric projection per fighter (minute-weighted mean absolute error, lower is better)\n");
    std::printf("  %-26s %10s %12s %12s\n", "metric", "matchup", "own career", "league avg");
    slpm.print("Sig strikes landed / min");
    acc.print("Sig strike accuracy");
    td.print("Takedowns / 15 min");
    ctrl.print("Control time share");
    return 0;
}

// ---------- profile ----------

// Career stats of established fighters, for percentiles.
struct Pool {
    std::vector<Stats> v;
    double pct(double Stats::*f, double x) const {
        double below = 0, eq = 0;
        for (auto& s : v) below += s.*f < x, eq += s.*f == x;
        return (below + eq / 2) / v.size();
    }
};

static std::string cm(double x) { return std::isnan(x) ? "?" : std::to_string((int)std::lround(x)) + " cm"; }

static int profile(const World& w, const Record& r) {
    const Stats& L = w.L;
    Stats s = stats(r, L);
    Pool P;
    for (auto& [_, o] : w.rec)
        if (o.n >= 5) P.v.push_back(stats(o, L));
    auto pc = [&](double Stats::*f) { return P.pct(f, s.*f); };

    std::printf("\n%s   UFC record %d-%d   Elo %.0f (peak %.0f)\n", r.name.c_str(), r.wins, r.n - r.wins, r.elo, r.peak_elo);
    std::printf("%s | %s stance | height %s | reach %s | last fight %s\n", r.weight_class.c_str(),
                r.stance.empty() ? "?" : r.stance.c_str(), cm(r.height).c_str(), cm(r.reach).c_str(), date_str(r.last_date).c_str());
    std::printf("Wins:   %d KO/TKO, %d Sub, %d Dec\nLosses: %d KO/TKO, %d Sub, %d Dec\n", r.win_by[KO], r.win_by[SUB],
                r.win_by[DEC], r.loss_by[KO], r.loss_by[SUB], r.loss_by[DEC]);

    struct Row { const char* label; double Stats::*f; bool percent; };
    static const Row rows[] = {
        {"Sig strikes landed / min", &Stats::slpm, false},     {"Sig strikes absorbed / min", &Stats::sapm, false},
        {"Sig strike accuracy", &Stats::sig_acc, true},        {"Sig strike defense", &Stats::str_def, true},
        {"Knockdowns / 15 min", &Stats::kd15, false},          {"Takedowns / 15 min", &Stats::td15, false},
        {"Takedown accuracy", &Stats::td_acc, true},           {"Takedown defense", &Stats::td_def, true},
        {"Submission attempts / 15 min", &Stats::sub15, false}, {"Control time share", &Stats::ctrl, true},
    };
    std::printf("\n  %-30s %9s %9s %6s\n", "Metric", "fighter", "league", "pctl*");
    for (auto& row : rows) {
        double k = row.percent ? 100 : 1;
        std::printf(row.percent ? "  %-30s %8.1f%% %8.1f%% %6.0f\n" : "  %-30s %9.2f %9.2f %6.0f\n", row.label,
                    k * (s.*row.f), k * (L.*row.f), 100 * pc(row.f));
    }
    std::printf("  * percentile among %zu fighters with 5+ UFC fights%s\n", P.v.size(),
                r.n < 5 ? "; small sample, so rates lean on league averages" : "");

    // Style: archetype from striking vs grappling percentiles, plus standout tags.
    double strike = (pc(&Stats::slpm) + pc(&Stats::kd15) + pc(&Stats::sig_acc) + pc(&Stats::sig_att_pm)) / 4;
    double grapple = (pc(&Stats::td15) + pc(&Stats::ctrl) + pc(&Stats::sub15)) / 3;
    const char* archetype = grapple - strike > 0.15
        ? (pc(&Stats::sub15) > std::max(pc(&Stats::td15), pc(&Stats::ctrl)) ? "Submission grappler" : "Wrestler")
        : strike - grapple > 0.15 ? "Striker" : "All-rounder";
    std::string tags;
    auto tag = [&](bool on, const char* t) { if (on) tags += (tags.empty() ? "" : ", ") + std::string(t); };
    tag(pc(&Stats::sig_att_pm) >= 0.8, "volume striker");
    tag(pc(&Stats::kd15) >= 0.8 || (r.wins >= 3 && 2 * r.win_by[KO] >= r.wins), "knockout power");
    tag(pc(&Stats::sig_acc) >= 0.8, "precision striker");
    tag(pc(&Stats::str_def) >= 0.8, "hard to hit");
    tag(pc(&Stats::slpm) >= 0.7 && pc(&Stats::sapm) >= 0.7, "brawler (trades heavily)");
    tag(pc(&Stats::td15) >= 0.8, "chain wrestler");
    tag(pc(&Stats::ctrl) >= 0.8, "top control");
    tag(pc(&Stats::sub15) >= 0.8 || (r.wins >= 3 && 2 * r.win_by[SUB] >= r.wins), "submission threat");
    tag(pc(&Stats::td_def) >= 0.8 && pc(&Stats::td15) <= 0.4, "sprawl-and-brawl");
    tag(pc(&Stats::rev15) >= 0.85, "scrambler");
    tag(pc(&Stats::kd_abs15) >= 0.8 || r.loss_by[KO] >= 3, "durability concerns");
    std::printf("\nStyle: %s%s%s\n", archetype, tags.empty() ? "" : " - ", tags.c_str());

    // Movement & habits: proxies only, the dataset has fight totals but no positional or per-round splits.
    std::printf("\nMovement & habits (inferred from fight totals; no positional/per-round data in this dataset)\n");
    double vol = pc(&Stats::sig_att_pm), def = pc(&Stats::str_def);
    const char* engage = vol >= 0.7 ? "pressure/volume - sets a high pace and keeps throwing"
                       : vol <= 0.35 && def >= 0.55 ? "counter-striker - low output, hard to hit, waits for openings"
                       : vol <= 0.35 ? "low output - fights in bursts"
                       : "measured - mid-pack output";
    std::printf("  Pace             %.1f sig strike attempts/min (pctl %.0f), %.1f total attempts/min\n", s.sig_att_pm,
                100 * vol, s.str_att_pm);
    std::printf("  Engagement       %s; strike differential %+.2f/min\n", engage, s.slpm - s.sapm);
    double neutral = std::max(0.0, 1 - s.ctrl - s.ctrl_abs);
    const char* where = s.ctrl >= 0.25 ? "top-position fighter"
                      : s.ctrl_abs >= 0.25 ? "spends long stretches controlled (bottom / against the cage)"
                      : neutral >= 0.8 ? "keeps it standing, works at range"
                      : "mixes standing and grappling exchanges";
    std::printf("  Position time    %.0f%% controlling, %.0f%% controlled, ~%.0f%% neutral -> %s\n", 100 * s.ctrl,
                100 * s.ctrl_abs, 100 * neutral, where);
    double close = pc(&Stats::nonsig_share);
    std::printf("  Close-range work %.0f%% of landed strikes are non-significant (pctl %.0f) -> %s\n", 100 * s.nonsig_share,
                100 * close, close >= 0.7 ? "heavy clinch / ground-and-pound work" : close <= 0.3 ? "mostly clean shots at range" : "balanced");
    std::printf("  Level changes    %.1f takedown attempts/15 min, %.0f%% land\n", s.td15 / s.td_acc, 100 * s.td_acc);
    std::printf("  Scrambles        %.2f reversals/15 min (pctl %.0f)\n", s.rev15, 100 * pc(&Stats::rev15));
    int fin = 0, dec = 0;
    double rounds = 0;
    for (auto& b : r.bouts) {
        if (b.method == DEC) ++dec;
        else if (b.won && b.method != OTHER) ++fin, rounds += b.round;
    }
    std::printf("  Fight length     %.1f min average, %.0f%% go to decision", r.secs / 60 / r.n, 100.0 * dec / r.n);
    if (fin) std::printf(", own finishes come in round %.1f on average", rounds / fin);
    std::printf("\n  Durability       %.2f knockdowns absorbed/15 min, finished %d time(s)\n", s.kd_abs15,
                r.loss_by[KO] + r.loss_by[SUB]);
    size_t k = std::min<size_t>(5, r.bouts.size()), from = r.bouts.size() - k;
    double sig = 0, secs = 0;
    for (size_t i = from; i < r.bouts.size(); ++i) sig += r.bouts[i].sig, secs += r.bouts[i].secs;
    double elo_before = from ? r.bouts[from - 1].elo_after : 1500;
    std::printf("  Recent form      last %zu: %.2f sig landed/min vs %.2f career; Elo %+.0f\n", k, sig / (secs / 60),
                r.sig / (r.secs / 60), r.elo - elo_before);

    std::printf("\nRecent fights\n");
    for (size_t i = r.bouts.size(); i-- > from;) {
        const Bout& b = r.bouts[i];
        std::printf("  %s  %c  %-10s R%d  vs %-24s sig %3.0f-%-3.0f TD %.0f\n", date_str(b.date).c_str(), b.won ? 'W' : 'L',
                    METHOD_NAME[b.method], b.round, b.opp.c_str(), b.sig, b.sig_abs, b.td);
    }
    return 0;
}

// ---------- predict ----------

static int predict(const World& w, const Record& a, const Record& b, int rounds) {
    if (&a == &b) return std::puts("Pick two different fighters."), 1;
    const Stats& L = w.L;
    Logit model = train(w, w.samples.size(), false);
    double p = model.prob(matchup(a, b, L, w.ref_date, w.ref_date, 0));
    Stats sa = stats(a, L), sb = stats(b, L);

    // Method split: blend the winner's win-method mix with the loser's loss-method mix, each shrunk to the league mix.
    auto mix = [&](const int* by, int m) {
        double league = w.all.win_by[m] / double(w.all.win_by[KO] + w.all.win_by[SUB] + w.all.win_by[DEC]);
        return (by[m] + 3 * league) / (by[KO] + by[SUB] + by[DEC] + 3);
    };
    double meth[2][3];
    for (int m = 0; m < 3; ++m) {
        meth[0][m] = p * (mix(a.win_by, m) + mix(b.loss_by, m)) / 2;
        meth[1][m] = (1 - p) * (mix(b.win_by, m) + mix(a.loss_by, m)) / 2;
    }
    double p_dec = meth[0][DEC] + meth[1][DEC], fin_secs = 0;
    int fin_n = 0;
    for (auto& f : w.fights)
        if (f.rounds == rounds && f.method != DEC) fin_secs += f.secs, ++fin_n;
    double minutes = p_dec * rounds * 5 + (1 - p_dec) * (fin_n ? fin_secs / fin_n / 60 : rounds * 2.5);

    double slpm[2] = {project(sa.slpm, sb.sapm, L.slpm), project(sb.slpm, sa.sapm, L.slpm)};
    double acc[2] = {project(sa.sig_acc, 1 - sb.str_def, L.sig_acc), project(sb.sig_acc, 1 - sa.str_def, L.sig_acc)};
    double td15[2] = {project(sa.td15, sb.td_abs15, L.td15), project(sb.td15, sa.td_abs15, L.td15)};
    double sub15[2] = {project(sa.sub15, sb.sub_abs15, L.sub15), project(sb.sub15, sa.sub_abs15, L.sub15)};
    double kd15[2] = {project(sa.kd15, sb.kd_abs15, L.kd15), project(sb.kd15, sa.kd_abs15, L.kd15)};
    double ctrl[2] = {project(sa.ctrl, sb.ctrl_abs, L.ctrl), project(sb.ctrl, sa.ctrl_abs, L.ctrl)};
    if (double t = ctrl[0] + ctrl[1]; t > 0.9) ctrl[0] *= 0.9 / t, ctrl[1] *= 0.9 / t;

    auto row = [](const char* label, double x, double y, const char* fmt) {
        char bx[32], by[32];
        std::snprintf(bx, sizeof bx, fmt, x), std::snprintf(by, sizeof by, fmt, y);
        std::printf("  %-32s %22s %22s\n", label, bx, by);
    };
    std::printf("\n%s vs %s  (%d rounds)\n\n  %-32s %22.22s %22.22s\n", a.name.c_str(), b.name.c_str(), rounds, "",
                a.name.c_str(), b.name.c_str());
    row("Win probability", 100 * p, 100 * (1 - p), "%.0f%%");
    row("  by KO/TKO", 100 * meth[0][KO], 100 * meth[1][KO], "%.0f%%");
    row("  by Submission", 100 * meth[0][SUB], 100 * meth[1][SUB], "%.0f%%");
    row("  by Decision", 100 * meth[0][DEC], 100 * meth[1][DEC], "%.0f%%");
    row("Sig strikes landed / min", slpm[0], slpm[1], "%.2f");
    row("Sig strike accuracy", 100 * acc[0], 100 * acc[1], "%.0f%%");
    row("Sig strikes landed (fight)", slpm[0] * minutes, slpm[1] * minutes, "%.0f");
    row("Takedowns / 15 min", td15[0], td15[1], "%.2f");
    row("Takedowns landed (fight)", td15[0] * minutes / 15, td15[1] * minutes / 15, "%.1f");
    row("Control time share", 100 * ctrl[0], 100 * ctrl[1], "%.0f%%");
    row("Submission attempts / 15 min", sub15[0], sub15[1], "%.2f");
    row("Chance of scoring a knockdown", 100 * (1 - std::exp(-kd15[0] * minutes / 15)),
        100 * (1 - std::exp(-kd15[1] * minutes / 15)), "%.0f%%");
    row("Elo", a.elo, b.elo, "%.0f");
    std::printf("\n  Expected fight length %.1f of %d min (decision %.0f%%)\n", minutes, rounds * 5, 100 * p_dec);
    for (const Record* r : {&a, &b})
        if (r->n < 3) std::printf("  Note: %s has only %d UFC fight(s); projections lean on league averages.\n", r->name.c_str(), r->n);
    std::printf("  Corner-blind logistic model trained on all %zu fights through %s; run `evaluate` for its backtest.\n",
                w.samples.size(), date_str(w.ref_date).c_str());
    return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string cmd = argc > 1 ? argv[1] : "";
    if (!(cmd == "evaluate" && argc == 2) && !(cmd == "profile" && argc == 3) && !(cmd == "predict" && (argc == 4 || argc == 5))) {
        std::puts("usage:\n  ufc evaluate\n  ufc profile \"<fighter>\"\n  ufc predict \"<fighter A>\" \"<fighter B>\" [3|5 rounds]");
        return 2;
    }
    int rounds = argc == 5 ? std::atoi(argv[4]) : 3;
    if (rounds != 3 && rounds != 5) return std::puts("rounds must be 3 or 5"), 2;
    try {
        World w = build(data_dir());
        if (cmd == "evaluate") return evaluate(w);
        if (cmd == "profile") {
            const Record* r = lookup(w, argv[2]);
            return r ? profile(w, *r) : 1;
        }
        const Record *a = lookup(w, argv[2]), *b = lookup(w, argv[3]);
        return a && b ? predict(w, *a, *b, rounds) : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
