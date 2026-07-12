#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <simdjson.h>

namespace agent::tools {

static double calc_factorial(double n) {
    if (n < 0 || n > 170 || n != std::floor(n))
        return std::numeric_limits<double>::quiet_NaN();
    double r = 1.0;
    for (int i = 2; i <= static_cast<int>(n); ++i)
        r *= i;
    return r;
}

static long long calc_gcd(long long a, long long b) {
    a = std::abs(a);
    b = std::abs(b);
    while (b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

static Result<std::string> calculator_callback(std::string_view arguments, void *) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(arguments.data(), arguments.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj))
        return fail<std::string>(ErrorCode::ParseError, "Expected JSON object");

    std::string_view op;
    double a = 0.0, b = 0.0;

    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        std::string_view kv = k.value();
        if (kv == "op") {
            field.value().get_string().get(op);
        } else if (kv == "a") {
            auto dv = field.value().get_double();
            if (!dv.error()) {
                a = dv.value();
            } else {
                int64_t iv;
                if (!field.value().get_int64().get(iv))
                    a = static_cast<double>(iv);
            }
        } else if (kv == "b") {
            auto dv = field.value().get_double();
            if (!dv.error()) {
                b = dv.value();
            } else {
                int64_t iv;
                if (!field.value().get_int64().get(iv))
                    b = static_cast<double>(iv);
            }
        }
    }

    if (op.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'op'");

    double r = 0.0;

    if (op == "+" || op == "add")
        r = a + b;
    else if (op == "-" || op == "sub")
        r = a - b;
    else if (op == "*" || op == "mul")
        r = a * b;
    else if (op == "/" || op == "div") {
        if (b == 0)
            return fail<std::string>(ErrorCode::InvalidConfig, "Division by zero");
        r = a / b;
    } else if (op == "%" || op == "mod") {
        if (b == 0)
            return fail<std::string>(ErrorCode::InvalidConfig, "Modulo by zero");
        r = std::fmod(a, b);
    } else if (op == "pow")
        r = std::pow(a, b);
    else if (op == "atan2")
        r = std::atan2(a, b);
    else if (op == "hypot")
        r = std::hypot(a, b);
    else if (op == "log_base") {
        if (b <= 0 || b == 1)
            return fail<std::string>(ErrorCode::InvalidConfig, "Invalid log base");
        r = std::log(a) / std::log(b);
    } else if (op == "gcd")
        r = static_cast<double>(calc_gcd(static_cast<long long>(a), static_cast<long long>(b)));
    else if (op == "lcm") {
        auto g = calc_gcd(static_cast<long long>(a), static_cast<long long>(b));
        r = g == 0 ? 0 : static_cast<double>(static_cast<long long>(a) / g * static_cast<long long>(b));
    } else if (op == "min")
        r = std::min(a, b);
    else if (op == "max")
        r = std::max(a, b);
    else if (op == "nCr")
        r = calc_factorial(a) / (calc_factorial(b) * calc_factorial(a - b));
    else if (op == "nPr")
        r = calc_factorial(a) / calc_factorial(a - b);
    else if (op == "sqrt")
        r = std::sqrt(a);
    else if (op == "cbrt")
        r = std::cbrt(a);
    else if (op == "abs")
        r = std::abs(a);
    else if (op == "floor")
        r = std::floor(a);
    else if (op == "ceil")
        r = std::ceil(a);
    else if (op == "round")
        r = std::round(a);
    else if (op == "trunc")
        r = std::trunc(a);
    else if (op == "sign")
        r = (a > 0) - (a < 0);
    else if (op == "square")
        r = a * a;
    else if (op == "cube")
        r = a * a * a;
    else if (op == "exp")
        r = std::exp(a);
    else if (op == "exp2")
        r = std::exp2(a);
    else if (op == "log")
        r = std::log(a);
    else if (op == "log2")
        r = std::log2(a);
    else if (op == "log10")
        r = std::log10(a);
    else if (op == "sin")
        r = std::sin(a);
    else if (op == "cos")
        r = std::cos(a);
    else if (op == "tan")
        r = std::tan(a);
    else if (op == "asin")
        r = std::asin(a);
    else if (op == "acos")
        r = std::acos(a);
    else if (op == "atan")
        r = std::atan(a);
    else if (op == "sinh")
        r = std::sinh(a);
    else if (op == "cosh")
        r = std::cosh(a);
    else if (op == "tanh")
        r = std::tanh(a);
    else if (op == "asinh")
        r = std::asinh(a);
    else if (op == "acosh")
        r = std::acosh(a);
    else if (op == "atanh")
        r = std::atanh(a);
    else if (op == "factorial")
        r = calc_factorial(a);
    else if (op == "deg2rad")
        r = a * M_PI / 180.0;
    else if (op == "rad2deg")
        r = a * 180.0 / M_PI;
    else
        return fail<std::string>(ErrorCode::InvalidConfig, "Unknown op: " + std::string(op));

    if (std::isnan(r))
        return ok(std::string("NaN"));
    if (std::isinf(r))
        return ok(r > 0 ? std::string("Inf") : std::string("-Inf"));

    if (r == std::floor(r) && std::abs(r) < 1e15) {
        return ok(std::to_string(static_cast<long long>(r)));
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.15g", r);
    return ok(std::string(buf));
}

Tool get_calculator_tool() {
    return Tool{
        .name = "calculator",
        .description =
            "Scientific calculator. Required: op (string), a (number). Optional: b (number) for binary ops.\n"
            "Binary ops: + - * / % pow atan2 hypot log_base gcd lcm min max nCr nPr\n"
            "Unary ops:  sqrt cbrt abs floor ceil round trunc sign square cube\n"
            "            exp exp2 log log2 log10\n"
            "            sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh\n"
            "            factorial deg2rad rad2deg",
        .parameter_schema =
            R"({"type":"object","properties":{"op":{"type":"string","description":"operation name"},"a":{"type":"number"},"b":{"type":"number","description":"second operand for binary ops"}},"required":["op","a"]})",
        .callback = calculator_callback,
        .user_data = nullptr,
    };
}

} // namespace agent::tools
