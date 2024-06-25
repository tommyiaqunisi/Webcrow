mysql -u "ai" "-pproverb" -e "CREATE DATABASE IF NOT EXISTS lexicon_en"
mysql -u "ai" "-pproverb" "lexicon_en" < "/home/iaquinta/webcrow/MySQLbackups/lexicon_en-tables/Words.sql"

mysql -u "ai" "-pproverb" -e "CREATE DATABASE IF NOT EXISTS termdf_en"
mysql -u "ai" "-pproverb" "termdf_en" < "/home/iaquinta/webcrow/MySQLbackups/termdf_en-tables/term_df.sql"
mysql -u "ai" "-pproverb" "termdf_en" < "/home/iaquinta/webcrow/MySQLbackups/termdf_en-tables/props.sql"

mysql -u "ai" "-pproverb" -e "CREATE DATABASE IF NOT EXISTS lexicon_np_en"
mysql -u "ai" "-pproverb" -e "CREATE DATABASE IF NOT EXISTS lexicon_pc_en"
mysql -u "ai" "-pproverb" -e "CREATE DATABASE IF NOT EXISTS lexicon_acr_abbr_en"

mysql -u "ai" "-pproverb" -e "CREATE DATABASE IF NOT EXISTS morph_en"
mysql -u "ai" "-pproverb" "morph_en" < "/home/iaquinta/webcrow/MySQLbackups/morph_en-tables/morph.sql"
