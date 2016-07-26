(defproject parapepoid "0.1.0-SNAPSHOT"
  :description "Colorz."
  :url "http://brutalism.rs"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.52.2"]
                 [net.mikera/vectorz-clj "0.44.0"]
                 [thi.ng/color "1.2.0"]
                 [thi.ng/geom "0.0.1173-SNAPSHOT"]
                 [incanter "1.5.7"]]
  :main ^:skip-aot parapepoid.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
