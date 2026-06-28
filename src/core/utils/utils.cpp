#include "utils.h"

namespace agent::utils {

void log(const Session &session, LogLevel level, std::string_view message) {
    if (session.config.logger) {
        session.config.logger(level, message, session.config.logger_user_data);
    }
}

} // namespace agent::utils