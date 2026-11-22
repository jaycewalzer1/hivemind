#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "search/agent.h"

inline std::vector<RootEdgeStats> selfplay_policy_edges(
    const std::vector<RootEdgeStats>& edges, NodeType rootType,
    const JointActionCandidate& choice, bool certifiedChoice) {
    if (certifiedChoice || rootType == NodeType::LOSS) {
        return {{choice, 1, 0.0f}};
    }
    const bool hasAlternative = std::any_of(edges.begin(), edges.end(), [](const RootEdgeStats& edge) {
        return edge.childType != NodeType::WIN;
    });
    std::vector<RootEdgeStats> eligible;
    for (const RootEdgeStats& edge : edges) {
        if (rootType == NodeType::WIN ? edge.childType == NodeType::LOSS
                                     : !hasAlternative || edge.childType != NodeType::WIN) {
            eligible.push_back(edge);
        }
    }
    if (eligible.empty()) {
        throw std::runtime_error("Solved root has no eligible policy edges");
    }
    if (std::none_of(eligible.begin(), eligible.end(), [](const RootEdgeStats& edge) {
            return edge.visits > 0;
        })) {
        for (RootEdgeStats& edge : eligible) {
            edge.visits = 1;
        }
    }
    return eligible;
}

/**
 * Spec v6 review fix: the policy target of a triggered hybrid search, which
 * never holds the root's hybrid sit (the action passing on both boards; the
 * sit plane says it is illegal here). The sit's edge is dropped before the
 * solved-root filter, and a proof that stood only through the sit (a root
 * WIN with no other lost child, or the sit as the certified or LOSS-root
 * choice) no longer narrows the target: the other edges are read as an
 * unsolved root's.
 */
inline std::vector<RootEdgeStats> hybrid_policy_edges(
    const std::vector<RootEdgeStats>& edges, NodeType rootType,
    const JointActionCandidate& choice, bool certifiedChoice) {
    const auto isSit = [](const JointActionCandidate& action) {
        return action.moveA == Stockfish::MOVE_NONE && action.moveB == Stockfish::MOVE_NONE;
    };
    std::vector<RootEdgeStats> moves;
    for (const RootEdgeStats& edge : edges) {
        if (!isSit(edge.action)) {
            moves.push_back(edge);
        }
    }
    if (moves.empty()) {
        return selfplay_policy_edges(edges, rootType, choice, certifiedChoice);
    }
    const bool sitChoice = isSit(choice);
    if (rootType == NodeType::WIN
        && std::none_of(moves.begin(), moves.end(), [](const RootEdgeStats& edge) {
               return edge.childType == NodeType::LOSS;
           })) {
        rootType = NodeType::UNSOLVED;
    }
    if (sitChoice && rootType == NodeType::LOSS) {
        rootType = NodeType::UNSOLVED;
    }
    return selfplay_policy_edges(moves, rootType, choice, certifiedChoice && !sitChoice);
}
