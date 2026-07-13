#include <agent-cpp/agent.hpp>

namespace agent {

namespace {

Result<std::string> data_source_tool_callback(std::string_view arguments, void *user_data) {
    auto *source = static_cast<DataSource *>(user_data);
    if (!source || !source->query)
        return fail<std::string>(ErrorCode::InvalidConfig, "DataSource has no query function.");
    return source->query(source->state.get(), arguments, source->user_data);
}

} // namespace

Result<void> register_data_source(Session &session, DataSource ds) {
    if (ds.name.empty())
        return fail(ErrorCode::InvalidConfig, "DataSource requires a non-empty name.");
    std::string key = ds.name;
    session.data_sources[key] = std::move(ds);
    return ok();
}

Result<std::string> query_data_source(Session &session, std::string_view name,
                                      std::string_view query) {
    auto it = session.data_sources.find(std::string(name));
    if (it == session.data_sources.end())
        return fail<std::string>(ErrorCode::KeyNotFound,
                                 "Data source not found: " + std::string(name));
    const DataSource &ds = it->second;
    if (!ds.query)
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "Data source has no query function: " + std::string(name));
    return ds.query(ds.state.get(), query, ds.user_data);
}

Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source,
                           std::string_view description, std::string_view parameter_schema) {
    session.retained_data_sources.push_back(source);
    Tool t;
    t.name = source->name;
    t.description = description;
    t.parameter_schema = parameter_schema;
    t.callback = data_source_tool_callback;
    t.user_data = source.get();
    return t;
}

} // namespace agent
