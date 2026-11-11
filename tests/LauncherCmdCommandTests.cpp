#include "cmdCommand.h"

#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QProcess>
#include <QString>
#include <QStringList>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr char kPrintArguments[] = "--print-arguments";
constexpr char kArgumentLine[] = "ARG ";

void Check(bool value, const QString &message) {
  if (!value) {
    std::fprintf(stderr, "LauncherCmdCommandTests: failed: %s\n",
                 qPrintable(message));
    std::abort();
  }
}

int PrintArguments(const QStringList &arguments) {
  for (const auto &argument : arguments) {
    std::printf("%s%s\n", kArgumentLine,
                argument.toUtf8().toHex().constData());
  }
  return 0;
}

QStringList ReceivedArguments(const QByteArray &output) {
  QStringList received;
  for (const auto &line : output.split('\n')) {
    if (line.startsWith(kArgumentLine)) {
      received.append(QString::fromUtf8(
          QByteArray::fromHex(line.mid(qstrlen(kArgumentLine)).trimmed())));
    }
  }
  return received;
}

QStringList OwnCommand() {
  return {QDir::toNativeSeparators(QCoreApplication::applicationFilePath()),
          QString::fromLatin1(kPrintArguments)};
}

QByteArray RunThroughCmd(const CmdLaunch &launch) {
  QProcess cmd;
  cmd.setProgram(QStringLiteral("cmd.exe"));
  cmd.setNativeArguments(launch.native_arguments);
  cmd.setProcessEnvironment(launch.environment);
  cmd.start();
  Check(cmd.waitForStarted(30000), QStringLiteral("cmd.exe did not start"));
  cmd.closeWriteChannel();
  Check(cmd.waitForFinished(30000), QStringLiteral("cmd.exe did not finish"));
  return cmd.readAllStandardOutput();
}

void CheckValuesArriveVerbatim(bool close_on_clean_exit) {
  const QStringList values = {
      QStringLiteral("a\"b"),
      QStringLiteral("trailing\\"),
      QStringLiteral("two\\\\"),
      QStringLiteral("!PATH! %PATH% ^! ^^ !!"),
      QStringLiteral("C:\\Games\\Some Game (1)\\eboot.bin"),
      QStringLiteral(u"\u00fcn\u00efc\u00f6d\u00e9 \u20ac \u65e5\u672c"),
      QString(),
      QStringLiteral(" leading and trailing "),
  };
  const auto launch = CmdCommand(OwnCommand() + values, close_on_clean_exit,
                                 QProcessEnvironment::systemEnvironment());
  Check(launch.has_value(), QStringLiteral("short values were refused"));
  const QStringList received = ReceivedArguments(RunThroughCmd(*launch));
  Check(received == values,
        QStringLiteral("arguments changed on the way: sent %1, received %2")
            .arg(values.size())
            .arg(received.size()));
}

void CheckLongestCommandArrives() {
  constexpr qsizetype separator_and_quotes = 3;
  const qsizetype longest = CmdArgumentsLimit - 2 * separator_and_quotes -
                            static_cast<qsizetype>(qstrlen(kPrintArguments));
  const auto environment = QProcessEnvironment::systemEnvironment();
  const QString fits(longest, u'a');
  const auto launch =
      CmdCommand(OwnCommand() + QStringList{fits}, true, environment);
  Check(launch.has_value(),
        QStringLiteral("a command at cmd.exe's limit was refused"));
  Check(ReceivedArguments(RunThroughCmd(*launch)) == QStringList{fits},
        QStringLiteral("a command at cmd.exe's limit did not arrive"));
  Check(!CmdCommand(OwnCommand() + QStringList{fits + u'a'}, true, environment)
             .has_value(),
        QStringLiteral("a command over cmd.exe's limit was accepted"));
  Check(!CmdCommand(OwnCommand() + QStringList{QString(longest, u'"')}, true,
                    environment)
             .has_value(),
        QStringLiteral("escaped quotes were not counted against the limit"));
  Check(!CmdCommand(OwnCommand() +
                        QStringList{QStringLiteral("a") + QChar(u'\0') +
                                    QStringLiteral("b")},
                    true, environment)
             .has_value(),
        QStringLiteral("a value with a NUL character was accepted"));
}

} // namespace

int main(int argc, char **argv) {
  QCoreApplication application(argc, argv);
  const QStringList own = QCoreApplication::arguments();
  if (own.size() > 1 && own[1] == QLatin1String(kPrintArguments)) {
    return PrintArguments(own.mid(2));
  }
  CheckValuesArriveVerbatim(false);
  CheckValuesArriveVerbatim(true);
  CheckLongestCommandArrives();
  std::printf("LauncherCmdCommandTests: ok\n");
  return 0;
}
