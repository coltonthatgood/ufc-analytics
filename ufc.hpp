// UFC analytics core: data loading, chronological fighter histories, Elo, logistic regression.
// Header-only so main.cpp and test.cpp share it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ufc {

// ---------- parsing ----------

inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quoted) {
            if (c != '"') cur += c;
            else if (i + 1 < line.size() && line[i + 1] == '"') cur += '"', ++i;
            else quoted = false;
        } else if (c == '"') quoted = true;
        else if (c == ',') out.push_back(cur), cur.clear();
        else if (c != '\r') cur += c;
    }
    out.push_back(cur);
    return out;
}

struct Table {
    std::vector<std::string> head;
    std::vector<std::vector<std::string>> rows;
    size_t col(const std::string& name) const {
        auto it = std::find(head.begin(), head.end(), name);
        if (it == head.end()) throw std::runtime_error("missing column: " + name);
        return it - head.begin();
    }
};

inline Table read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    Table t;
    std::string line;
    std::getline(in, line);
    if (line.rfind("\xEF\xBB\xBF", 0) == 0) line.erase(0, 3);
    t.head = split_csv(line);
    while (std::getline(in, line))
        if (!line.empty()) t.rows.push_back(split_csv(line));
    return t;
}

inline double num(const std::string& s) {
    char* end;
    double v = std::strtod(s.c_str(), &end);
    return end == s.c_str() ? NAN : v;
}

// Days since 1970-01-01 (Howard Hinnant's civil calendar algorithms).
inline int days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    int era = (y >= 0 ? y : y - 399) / 400, yoe = y - era * 400;
    int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    return era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
}

inline std::string date_str(int z) {
    z += 719468;
    int era = (z >= 0 ? z : z - 146096) / 146097, doe = z - era * 146097;
    int yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153;
    int d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9;
    char buf[16];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02d", yoe + era * 400 + (m <= 2), m, d);
    return buf;
}

// "March 23, 2024" -> days since epoch, -1 if unparseable.
inline int parse_date(const std::string& s) {
    static const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    std::istringstream in(s);
    std::string mon;
    int d = 0, y = 0;
    char comma;
    if (!(in >> mon >> d >> comma >> y)) return -1;
    for (int m = 0; m < 12; ++m)
        if (mon.compare(0, 3, months[m]) == 0) return days_from_civil(y, m + 1, d);
    return -1;
}

// ---------- fights ----------

enum Method { KO, SUB, DEC, OTHER };
inline const char* METHOD_NAME[] = {"KO/TKO", "Submission", "Decision", "DQ/Other"};

inline Method method_of(const std::string& m) {
    if (m.find("KO") != std::string::npos) return KO;  // includes "TKO - Doctor's Stoppage"
    if (m.find("Submission") != std::string::npos) return SUB;
    if (m.find("Decision") != std::string::npos) return DEC;
    return OTHER;
}

struct Corner {
    std::string name, stance;
    double kd, sig, sig_att, str, str_att, td, td_att, sub, rev, ctrl;  // this fight's totals
    double age, height, reach;  // age is as of the dataset scrape; NaN when missing
};

struct Fight {
    std::string event, weight_class;
    int date, rounds, finish_round, winner;  // winner: 0 = red corner, 1 = blue
    double secs;
    Method method;
    Corner c[2];
};

// Fights oldest-first. Dates come from joining event names to completed_events_small.csv.
inline std::vector<Fight> load_fights(const std::string& dir) {
    Table ev = read_csv(dir + "/completed_events_small.csv");
    std::unordered_map<std::string, int> date_of;
    for (auto& r : ev.rows) date_of[r[ev.col("event")]] = parse_date(r[ev.col("date")]);

    Table t = read_csv(dir + "/large_dataset.csv");
    const char* fields[] = {"kd", "sig_str", "sig_str_att", "str", "str_att", "td", "td_att",
                            "sub_att", "rev", "ctrl_sec", "age", "height", "reach"};
    size_t idx[2][13], name[2], stance[2];
    for (int s = 0; s < 2; ++s) {
        std::string p = s ? "b_" : "r_";
        for (int k = 0; k < 13; ++k) idx[s][k] = t.col(p + fields[k]);
        name[s] = t.col(p + "fighter");
        stance[s] = t.col(p + "stance");
    }
    size_t c_event = t.col("event_name"), c_win = t.col("winner"), c_wc = t.col("weight_class"),
           c_method = t.col("method"), c_fr = t.col("finish_round"), c_tr = t.col("total_rounds"),
           c_time = t.col("time_sec");

    std::vector<Fight> fights;
    for (auto it = t.rows.rbegin(); it != t.rows.rend(); ++it) {  // file is newest-first
        auto& r = *it;
        if (r.size() != t.head.size()) continue;
        auto d = date_of.find(r[c_event]);
        if (d == date_of.end() || d->second < 0 || (r[c_win] != "Red" && r[c_win] != "Blue")) continue;
        Fight f;
        f.event = r[c_event];
        f.weight_class = r[c_wc];
        f.date = d->second;
        f.winner = r[c_win] == "Blue";
        f.method = method_of(r[c_method]);
        f.finish_round = std::max(1, (int)num(r[c_fr]));
        double tr = num(r[c_tr]);
        f.rounds = std::isnan(tr) ? f.finish_round : (int)tr;
        f.secs = std::max(1.0, (f.finish_round - 1) * 300 + num(r[c_time]));
        for (int s = 0; s < 2; ++s) {
            double v[13];
            for (int k = 0; k < 13; ++k) {
                v[k] = num(r[idx[s][k]]);
                if (k < 10 && std::isnan(v[k])) v[k] = 0;
            }
            f.c[s] = {r[name[s]], r[stance[s]], v[0], v[1], v[2], v[3], v[4], v[5], v[6],
                      v[7], v[8], v[9], v[10], v[11], v[12]};
        }
        fights.push_back(std::move(f));
    }
    std::stable_sort(fights.begin(), fights.end(), [](const Fight& a, const Fight& b) { return a.date < b.date; });
    return fights;
}

// ---------- fighter history ----------

struct Bout {
    int date;
    std::string opp;
    bool won;
    Method method;
    int round;
    double secs, sig, sig_abs, td, elo_after;
};

// Career totals accumulated fight by fight, so at any point it only knows the past.
// ponytail: fighters are keyed by name (dataset has no IDs), so namesakes like the two Bruno Silvas merge.
struct Record {
    std::string name, stance, weight_class;
    double age = NAN, height = NAN, reach = NAN;
    int n = 0, wins = 0, streak = 0, last_date = -1;
    double secs = 0, kd = 0, sig = 0, sig_att = 0, str = 0, str_att = 0, td = 0, td_att = 0, sub = 0, rev = 0, ctrl = 0;
    double kd_abs = 0, sig_abs = 0, sig_faced = 0, td_abs = 0, td_faced = 0, sub_abs = 0, ctrl_abs = 0;
    int win_by[4] = {}, loss_by[4] = {};
    double elo = 1500, peak_elo = 1500;
    std::vector<Bout> bouts;

    // Static attributes are known before the fight, so set them before computing pre-fight features.
    void see(const Corner& c) {
        name = c.name;
        if (!c.stance.empty()) stance = c.stance;
        if (!std::isnan(c.age)) age = c.age;
        if (!std::isnan(c.height)) height = c.height;
        if (!std::isnan(c.reach)) reach = c.reach;
    }

    void add(const Corner& me, const Corner& op, const Fight& f, bool won) {
        ++n;
        wins += won;
        streak = won ? std::max(streak, 0) + 1 : std::min(streak, 0) - 1;
        last_date = f.date;
        weight_class = f.weight_class;
        secs += f.secs;
        kd += me.kd, sig += me.sig, sig_att += me.sig_att, str += me.str, str_att += me.str_att;
        td += me.td, td_att += me.td_att, sub += me.sub, rev += me.rev, ctrl += me.ctrl;
        kd_abs += op.kd, sig_abs += op.sig, sig_faced += op.sig_att, td_abs += op.td;
        td_faced += op.td_att, sub_abs += op.sub, ctrl_abs += op.ctrl;
        (won ? win_by : loss_by)[f.method]++;
        bouts.push_back({f.date, op.name, won, f.method, f.finish_round, f.secs, me.sig, op.sig, me.td, 0});
    }
};

inline double elo_expect(double a, double b) { return 1 / (1 + std::pow(10, (b - a) / 400)); }

inline void elo_update(Record& winner, Record& loser, double k = 32) {
    double d = k * (1 - elo_expect(winner.elo, loser.elo));
    winner.elo += d, loser.elo -= d;
    winner.peak_elo = std::max(winner.peak_elo, winner.elo);
}

// ---------- rates ----------

struct Stats {
    double slpm, sapm, sig_acc, str_def, sig_att_pm, str_att_pm, nonsig_share;
    double td15, td_acc, td_def, td_abs15, sub15, sub_abs15, kd15, kd_abs15, rev15;
    double ctrl, ctrl_abs, win_pct, finish_rate, finished_rate;
};

// Career rates shrunk toward league prior L by pseudo-counts (15 min, 30 strikes, 8 TD attempts, 3 fights).
// k scales the pseudo-counts; k = 0 gives raw ratios.
inline Stats stats(const Record& r, const Stats& L, double k = 1) {
    const double KM = 15 * k, KA = 30 * k, KT = 8 * k, KF = 3 * k, m = r.secs / 60;
    auto pm = [&](double x, double prior) { return (x + prior * KM) / (m + KM); };
    auto p15 = [&](double x, double prior) { return 15 * pm(x, prior / 15); };
    auto ratio = [](double x, double d, double prior, double K) { return (x + prior * K) / (d + K); };
    Stats s;
    s.slpm = pm(r.sig, L.slpm);
    s.sapm = pm(r.sig_abs, L.sapm);
    s.sig_acc = ratio(r.sig, r.sig_att, L.sig_acc, KA);
    s.str_def = 1 - ratio(r.sig_abs, r.sig_faced, 1 - L.str_def, KA);
    s.sig_att_pm = pm(r.sig_att, L.sig_att_pm);
    s.str_att_pm = pm(r.str_att, L.str_att_pm);
    s.nonsig_share = ratio(r.str - r.sig, r.str, L.nonsig_share, KA);
    s.td15 = p15(r.td, L.td15);
    s.td_acc = ratio(r.td, r.td_att, L.td_acc, KT);
    s.td_def = 1 - ratio(r.td_abs, r.td_faced, 1 - L.td_def, KT);
    s.td_abs15 = p15(r.td_abs, L.td_abs15);
    s.sub15 = p15(r.sub, L.sub15);
    s.sub_abs15 = p15(r.sub_abs, L.sub_abs15);
    s.kd15 = p15(r.kd, L.kd15);
    s.kd_abs15 = p15(r.kd_abs, L.kd_abs15);
    s.rev15 = p15(r.rev, L.rev15);
    s.ctrl = ratio(r.ctrl, r.secs, L.ctrl, KM * 60);
    s.ctrl_abs = ratio(r.ctrl_abs, r.secs, L.ctrl_abs, KM * 60);
    s.win_pct = ratio(r.wins, r.n, L.win_pct, KF);
    s.finish_rate = ratio(r.win_by[KO] + r.win_by[SUB], r.n, L.finish_rate, KF);
    s.finished_rate = ratio(r.loss_by[KO] + r.loss_by[SUB], r.n, L.finished_rate, KF);
    return s;
}

// Matchup projection: attacker's rate scaled by how much the defender concedes vs league average.
// Half-strength (sqrt) adjustment: beat both the full ratio and no adjustment on every metric in `evaluate`.
inline double project(double off, double def, double league) { return off * std::sqrt(def / league); }

// ---------- winner model ----------

inline const std::vector<std::string> FEATURES = {
    "elo", "slpm", "sapm", "sig_acc", "str_def", "td15", "td_acc", "td_def", "sub15", "kd15", "ctrl",
    "win_pct", "experience", "age", "reach", "height", "streak", "layoff", "finish_rate", "finished_rate", "red_corner"};

inline std::vector<double> fighter_vec(const Record& r, const Stats& L, int date, int ref_date) {
    Stats s = stats(r, L);
    double layoff = r.last_date < 0 ? 365 : std::min(1500, date - r.last_date);
    return {r.elo / 100, s.slpm, s.sapm, s.sig_acc, s.str_def, s.td15, s.td_acc, s.td_def, s.sub15, s.kd15, s.ctrl,
            s.win_pct, std::log1p(r.n), r.age - (ref_date - date) / 365.25, r.reach, r.height, double(r.streak),
            std::log1p(layoff), s.finish_rate, s.finished_rate};
}

// A-minus-B feature differences; a physical missing on either side contributes 0.
// corner: +1 if A is red, -1 if blue, 0 if unknown.
inline std::vector<double> matchup(const Record& a, const Record& b, const Stats& L, int date, int ref_date, double corner) {
    auto x = fighter_vec(a, L, date, ref_date), y = fighter_vec(b, L, date, ref_date);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::isnan(x[i]) || std::isnan(y[i]) ? 0 : x[i] - y[i];
    x.push_back(corner);
    return x;
}

// L2 logistic regression by gradient descent. Each sample is mirrored, (x, y) -> (-x, 1 - y), and there is
// no intercept or centering, so p(A beats B) = 1 - p(B beats A) exactly.
struct Logit {
    std::vector<double> w, sd;

    double prob(const std::vector<double>& x) const {
        double z = 0;
        for (size_t i = 0; i < w.size(); ++i) z += w[i] * x[i] / sd[i];
        return 1 / (1 + std::exp(-z));
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double l2 = 1e-3,
             int iters = 500, double lr = 0.5) {
        size_t n = X.size(), d = X[0].size();
        w.assign(d, 0), sd.assign(d, 0);
        for (auto& x : X)
            for (size_t i = 0; i < d; ++i) sd[i] += x[i] * x[i];
        for (auto& s : sd) s = s > 0 ? std::sqrt(s / n) : 1;  // RMS; mean is 0 after mirroring
        std::vector<double> g(d);
        for (int it = 0; it < iters; ++it) {
            std::fill(g.begin(), g.end(), 0);
            for (size_t k = 0; k < n; ++k) {
                // the mirrored copy has error -e on -x: identical gradient, so the mean over 2n equals this one
                double e = prob(X[k]) - y[k];
                for (size_t i = 0; i < d; ++i) g[i] += e * X[k][i] / sd[i];
            }
            for (size_t i = 0; i < d; ++i) w[i] -= lr * (g[i] / n + l2 * w[i]);
        }
    }
};

// ---------- world: one chronological pass over every fight ----------

struct Sample {
    std::vector<double> x;  // red-minus-blue pre-fight features
    int y;                  // 1 if red won
    size_t fight;
    int n_red, n_blue;      // prior UFC fights
    Stats red, blue;        // pre-fight career stats, for metric projection
};

struct World {
    std::vector<Fight> fights;
    Record all;  // league aggregate
    Stats L;
    int ref_date;  // last event in the data, the date ages refer to
    std::unordered_map<std::string, Record> rec;
    std::vector<Sample> samples;
};

inline World build(const std::string& dir) {
    World w;
    w.fights = load_fights(dir);
    if (w.fights.empty()) throw std::runtime_error("no fights loaded from " + dir);
    for (auto& f : w.fights) w.all.add(f.c[0], f.c[1], f, f.winner == 0), w.all.add(f.c[1], f.c[0], f, f.winner == 1);
    w.L = stats(w.all, Stats{}, 0);  // ponytail: league prior uses all years; a per-date prior if leakage matters
    w.ref_date = w.fights.back().date;
    for (size_t i = 0; i < w.fights.size(); ++i) {
        const Fight& f = w.fights[i];
        Record &r = w.rec[f.c[0].name], &b = w.rec[f.c[1].name];
        r.see(f.c[0]), b.see(f.c[1]);
        w.samples.push_back({matchup(r, b, w.L, f.date, w.ref_date, 1), f.winner == 0, i, r.n, b.n, stats(r, w.L), stats(b, w.L)});
        r.add(f.c[0], f.c[1], f, f.winner == 0);
        b.add(f.c[1], f.c[0], f, f.winner == 1);
        f.winner == 0 ? elo_update(r, b) : elo_update(b, r);
        r.bouts.back().elo_after = r.elo, b.bouts.back().elo_after = b.elo;
    }
    return w;
}

}  // namespace ufc
