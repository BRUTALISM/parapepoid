(ns parapepoid.core
  (:require [clojure.core.matrix :as matrix]))

(matrix/set-current-implementation :vectorz)

(defn- main []
  "Something something neural networks."
  [& args]
  (prn "OHAI"))

