(defproject parapepoid "0.1.0-SNAPSHOT"
  :description "Colorz."
  :url "http://brutalism.rs"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.52.2"]
                 [net.mikera/vectorz-clj "0.44.0"]]
  :main ^:skip-aot parapepoid.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
