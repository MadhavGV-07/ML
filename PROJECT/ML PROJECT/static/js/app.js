/**
 * SPECTRA AML // Graph Neural Network Anti-Money Laundering Frontend Controller
 * Powers interactive Vis.js graph network, Chart.js metric telemetry,
 * real-time multi-model GNN inference (GAT, GCN, GraphSAGE, Baseline MLP),
 * explainable AI (XAI) attention tracing, dynamic transaction ledgers, and FinCEN SAR export.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    network: null,
    nodesDataset: null,
    edgesDataset: null,
    rawNodes: [],
    rawEdges: [],
    activeModel: 'GAT',
    selectedNodeId: null,
    selectedEdge: null,
    currentThreshold: 0.50,
    physicsEnabled: true,
    trainingChart: null,
    benchmarkChart: null,
    currentSarData: null,
    activeDossierTab: 'profile'
  };

  // DOM Elements
  const el = {
    // Nav & KPIs
    systemStatus: document.getElementById('systemStatus'),
    kpiAccounts: document.getElementById('kpiAccounts'),
    kpiTransactions: document.getElementById('kpiTransactions'),
    kpiIllicit: document.getElementById('kpiIllicit'),
    kpiAuc: document.getElementById('kpiAuc'),
    kpiModel: document.getElementById('kpiModel'),
    btnRegenGraph: document.getElementById('btnRegenGraph'),
    btnBenchmarkModal: document.getElementById('btnBenchmarkModal'),
    btnExplorerModal: document.getElementById('btnExplorerModal'),
    btnUploadCsv: document.getElementById('btnUploadCsv'),

    // Left Panel
    modelCards: document.querySelectorAll('.model-card'),
    epochsInput: document.getElementById('epochsInput'),
    epochsVal: document.getElementById('epochsVal'),
    lrInput: document.getElementById('lrInput'),
    lrVal: document.getElementById('lrVal'),
    lossTypeSelect: document.getElementById('lossTypeSelect'),
    btnTrainModel: document.getElementById('btnTrainModel'),
    typologyFilterContainer: document.getElementById('typologyFilterContainer'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdVal: document.getElementById('thresholdVal'),

    // Canvas & Viewport
    graphContainer: document.getElementById('graphNetworkCanvas'),
    btnZoomIn: document.getElementById('btnZoomIn'),
    btnZoomOut: document.getElementById('btnZoomOut'),
    btnFitGraph: document.getElementById('btnFitGraph'),
    btnTogglePhysics: document.getElementById('btnTogglePhysics'),
    btnResetView: document.getElementById('btnResetView'),

    // Right Panel Dossier & Tabs
    dossierTabs: document.getElementById('dossierTabs'),
    tabBtnProfile: document.getElementById('tabBtnProfile'),
    tabBtnLedger: document.getElementById('tabBtnLedger'),
    dossierTabProfile: document.getElementById('dossierTabProfile'),
    dossierTabLedger: document.getElementById('dossierTabLedger'),
    dossierTxnCount: document.getElementById('dossierTxnCount'),
    accountLedgerBody: document.getElementById('accountLedgerBody'),
    dossierEmptyState: document.getElementById('dossierEmptyState'),
    dossierContent: document.getElementById('dossierContent'),
    dossierStatusBadge: document.getElementById('dossierStatusBadge'),
    dossierAccountId: document.getElementById('dossierAccountId'),
    dossierEntityType: document.getElementById('dossierEntityType'),
    dossierRiskNumber: document.getElementById('dossierRiskNumber'),
    dossierGaugeFill: document.getElementById('dossierGaugeFill'),
    dossierTypologyName: document.getElementById('dossierTypologyName'),
    dossierTypologyDetail: document.getElementById('dossierTypologyDetail'),
    dossierInflow: document.getElementById('dossierInflow'),
    dossierOutflow: document.getElementById('dossierOutflow'),
    dossierInDeg: document.getElementById('dossierInDeg'),
    dossierOutDeg: document.getElementById('dossierOutDeg'),
    dossierBetweenness: document.getElementById('dossierBetweenness'),
    dossierPageRank: document.getElementById('dossierPageRank'),
    dossierXaiHeader: document.getElementById('dossierXaiHeader'),
    dossierXaiBadge: document.getElementById('dossierXaiBadge'),
    dossierAttentionList: document.getElementById('dossierAttentionList'),
    btnExportSAR: document.getElementById('btnExportSAR'),

    // Bottom Drawer
    bottomDrawer: document.querySelector('.bottom-drawer'),
    toggleDrawer: document.getElementById('toggleDrawer'),
    drawerSummaryMetrics: document.getElementById('drawerSummaryMetrics'),
    cmTN: document.getElementById('cmTN'),
    cmFP: document.getElementById('cmFP'),
    cmFN: document.getElementById('cmFN'),
    cmTP: document.getElementById('cmTP'),
    statPrec: document.getElementById('statPrec'),
    statRecall: document.getElementById('statRecall'),
    statF1: document.getElementById('statF1'),
    statPrAuc: document.getElementById('statPrAuc'),

    // Edge Inspector Modal
    edgeModal: document.getElementById('edgeModal'),
    btnCloseEdgeModal: document.getElementById('btnCloseEdgeModal'),
    btnCloseEdgeFooter: document.getElementById('btnCloseEdgeFooter'),
    edgeModalBody: document.getElementById('edgeModalBody'),
    btnFocusSender: document.getElementById('btnFocusSender'),
    btnFocusReceiver: document.getElementById('btnFocusReceiver'),

    // Global Transaction Explorer Modal
    explorerModal: document.getElementById('explorerModal'),
    btnCloseExplorer: document.getElementById('btnCloseExplorer'),
    explorerSearchInput: document.getElementById('explorerSearchInput'),
    explorerIllicitOnly: document.getElementById('explorerIllicitOnly'),
    explorerTxnCount: document.getElementById('explorerTxnCount'),
    explorerTableBody: document.getElementById('explorerTableBody'),

    // Modals & SAML-D
    benchmarkModal: document.getElementById('benchmarkModal'),
    btnCloseBenchmark: document.getElementById('btnCloseBenchmark'),
    benchmarkTableBody: document.getElementById('benchmarkTableBody'),
    sarModal: document.getElementById('sarModal'),
    btnCloseSar: document.getElementById('btnCloseSar'),
    sarJsonContent: document.getElementById('sarJsonContent'),
    btnCopySar: document.getElementById('btnCopySar'),
    btnDownloadSarFile: document.getElementById('btnDownloadSarFile'),
    btnLoadLocalSaml: document.getElementById('btnLoadLocalSaml'),
    samlSampleSize: document.getElementById('samlSampleSize'),
    customSampleSize: document.getElementById('customSampleSize')
  };

  // =========================================================================
  // Initialize Application
  // =========================================================================
  initEventListeners();
  initTrainingChart();
  loadGraphData();

  function initEventListeners() {
    // Multi-Architecture Switcher (GAT, GCN, GraphSAGE, Baseline MLP)
    el.modelCards.forEach(card => {
      card.addEventListener('click', async () => {
        el.modelCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        state.activeModel = card.getAttribute('data-model');
        el.kpiModel.textContent = state.activeModel;

        setStatus(`SWITCHING TO ${state.activeModel}...`, true);
        await updatePredictionThreshold(state.currentThreshold);
        if (state.selectedNodeId !== null) {
          await handleNodeSelected(state.selectedNodeId);
        }
        setStatus(`ACTIVE ARCH: ${state.activeModel}`, false);
      });
    });

    // Hyperparameter Sliders
    el.epochsInput.addEventListener('input', (e) => el.epochsVal.textContent = e.target.value);
    el.lrInput.addEventListener('input', (e) => el.lrVal.textContent = e.target.value);
    
    el.thresholdSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      el.thresholdVal.textContent = val.toFixed(2);
      state.currentThreshold = val;
      updatePredictionThreshold(val);
    });

    // Train button
    el.btnTrainModel.addEventListener('click', handleTrainModel);

    // Canvas tools with smooth easing animations
    el.btnZoomIn.addEventListener('click', () => {
      if (state.network) {
        state.network.moveTo({ scale: state.network.getScale() * 1.35, animation: { duration: 320, easingFunction: 'easeInOutQuad' } });
      }
    });
    el.btnZoomOut.addEventListener('click', () => {
      if (state.network) {
        state.network.moveTo({ scale: state.network.getScale() * 0.72, animation: { duration: 320, easingFunction: 'easeInOutQuad' } });
      }
    });
    el.btnFitGraph.addEventListener('click', () => {
      if (state.network) {
        state.network.fit({ animation: { duration: 600, easingFunction: 'easeInOutCubic' } });
      }
    });
    el.btnTogglePhysics.addEventListener('click', () => {
      state.physicsEnabled = !state.physicsEnabled;
      if (state.network) state.network.setOptions({ physics: { enabled: state.physicsEnabled } });
      el.btnTogglePhysics.style.color = state.physicsEnabled ? 'var(--neon-cyan)' : 'var(--text-dim)';
      el.btnTogglePhysics.title = state.physicsEnabled ? 'Physics Active (Click to Pause)' : 'Physics Paused (Click to Float)';
    });
    el.btnResetView.addEventListener('click', resetGraphHighlight);

    // Dossier Tabs (Profile vs Transactions)
    el.tabBtnProfile.addEventListener('click', () => switchDossierTab('profile'));
    el.tabBtnLedger.addEventListener('click', () => switchDossierTab('ledger'));

    // Drawer toggle
    el.toggleDrawer.addEventListener('click', () => {
      el.bottomDrawer.classList.toggle('open');
    });

    // SAR Action
    el.btnExportSAR.addEventListener('click', handleExportSAR);

    // Edge Inspector Close
    el.btnCloseEdgeModal.addEventListener('click', () => el.edgeModal.style.display = 'none');
    el.btnCloseEdgeFooter.addEventListener('click', () => el.edgeModal.style.display = 'none');
    el.btnFocusSender.addEventListener('click', () => {
      if (state.selectedEdge) {
        el.edgeModal.style.display = 'none';
        handleNodeSelected(state.selectedEdge.from);
      }
    });
    el.btnFocusReceiver.addEventListener('click', () => {
      if (state.selectedEdge) {
        el.edgeModal.style.display = 'none';
        handleNodeSelected(state.selectedEdge.to);
      }
    });

    // Benchmarks modal
    el.btnBenchmarkModal.addEventListener('click', handleOpenBenchmark);
    el.btnCloseBenchmark.addEventListener('click', () => el.benchmarkModal.style.display = 'none');
    el.btnCloseSar.addEventListener('click', () => el.sarModal.style.display = 'none');

    // Global Transaction Explorer Modal
    el.btnExplorerModal.addEventListener('click', handleOpenExplorer);
    el.btnCloseExplorer.addEventListener('click', () => el.explorerModal.style.display = 'none');
    el.explorerSearchInput.addEventListener('input', debounce(loadExplorerTransactions, 300));
    el.explorerIllicitOnly.addEventListener('change', loadExplorerTransactions);

    // SAR Modal Buttons
    el.btnCopySar.addEventListener('click', () => {
      if (state.currentSarData) {
        navigator.clipboard.writeText(JSON.stringify(state.currentSarData, null, 2));
        el.btnCopySar.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        setTimeout(() => el.btnCopySar.innerHTML = '<i class="fa-solid fa-copy"></i> Copy JSON', 2000);
      }
    });
    el.btnDownloadSarFile.addEventListener('click', () => {
      if (state.selectedNodeId !== null) {
        window.location.href = `/api/sar/${state.selectedNodeId}?download=true`;
      }
    });

    // Regenerate Graph
    el.btnRegenGraph.addEventListener('click', handleRegenerateGraph);

    // Upload Dataset Modal
    el.btnUploadCsv.addEventListener('click', openUploadModal);
    document.getElementById('btnCloseUpload').addEventListener('click', closeUploadModal);
    document.getElementById('btnCancelUpload').addEventListener('click', closeUploadModal);
    document.getElementById('btnBrowseFile').addEventListener('click', () => document.getElementById('csvFileInput').click());
    document.getElementById('csvFileInput').addEventListener('change', handleFileSelected);
    document.getElementById('btnImportDataset').addEventListener('click', handleImportDataset);

    // Instant Load Local SAML-D Button
    el.btnLoadLocalSaml.addEventListener('click', handleLoadLocalSaml);

    // Drag-and-drop on drop zone
    const dz = document.getElementById('dropZone');
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) handleFileSelected({ target: { files: [f] } });
    });
  }

  function switchDossierTab(tab) {
    state.activeDossierTab = tab;
    if (tab === 'profile') {
      el.tabBtnProfile.classList.add('active');
      el.tabBtnLedger.classList.remove('active');
      el.dossierTabProfile.style.display = 'block';
      el.dossierTabLedger.style.display = 'none';
    } else {
      el.tabBtnLedger.classList.add('active');
      el.tabBtnProfile.classList.remove('active');
      el.dossierTabProfile.style.display = 'none';
      el.dossierTabLedger.style.display = 'block';
    }
  }

  // =========================================================================
  // Graph Network Rendering (Vis.js)
  // =========================================================================
  async function loadGraphData() {
    setStatus('FETCHING GRAPH TOPOLOGY...', true);
    try {
      const resp = await fetch('/api/graph');
      const data = await resp.json();
      state.rawNodes = data.nodes || [];
      state.rawEdges = data.edges || [];

      // Update KPIs
      el.kpiAccounts.textContent = data.num_nodes || state.rawNodes.length;
      el.kpiTransactions.textContent = data.num_edges || state.rawEdges.length;
      el.kpiIllicit.textContent = data.num_illicit || 0;
      el.kpiModel.textContent = data.active_model || state.activeModel;

      renderVisNetwork(state.rawNodes, state.rawEdges);
      buildDynamicTypologyFilters(state.rawNodes, state.rawEdges);
      setStatus('ENGINE READY', false);

      // Trigger initial prediction update to sync threshold & metrics
      await updatePredictionThreshold(state.currentThreshold);
    } catch (err) {
      console.error('Error loading graph:', err);
      setStatus('ERROR LOADING TOPOLOGY', false);
    }
  }

  function buildDynamicTypologyFilters(nodes, edges) {
    const typologies = new Set();
    nodes.forEach(n => {
      if (n.typology && n.typology !== 'Normal' && n.typology !== 'Licit Retail') {
        typologies.add(n.typology);
      }
    });
    edges.forEach(e => {
      if (e.typology && e.typology !== 'Normal' && e.typology !== 'Uploaded') {
        typologies.add(e.typology);
      }
    });

    el.typologyFilterContainer.innerHTML = '';
    
    // Always include "All Accounts"
    const allBtn = document.createElement('button');
    allBtn.className = 'filter-btn active';
    allBtn.setAttribute('data-typology', 'all');
    allBtn.textContent = 'All Accounts';
    el.typologyFilterContainer.appendChild(allBtn);

    // Always include "Illicit Alerts Only"
    const illBtn = document.createElement('button');
    illBtn.className = 'filter-btn alert-filter';
    illBtn.setAttribute('data-typology', 'Illicit');
    illBtn.textContent = 'Illicit Alerts Only';
    el.typologyFilterContainer.appendChild(illBtn);

    // Add unique dataset typologies (up to 5 most common)
    Array.from(typologies).slice(0, 5).forEach(typ => {
      const btn = document.createElement('button');
      btn.className = 'filter-btn';
      btn.setAttribute('data-typology', typ);
      btn.textContent = typ.replace(/_/g, ' ');
      el.typologyFilterContainer.appendChild(btn);
    });

    // Rebind filter buttons
    el.typologyFilterContainer.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        el.typologyFilterContainer.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyTypologyFilter(btn.getAttribute('data-typology'));
      });
    });
  }

  function renderVisNetwork(nodesData, edgesData) {
    // Format nodes for Vis.js
    const formattedNodes = nodesData.map(n => {
      const isIllicit = n.pred_is_illicit !== undefined ? (n.pred_is_illicit === 1) : (n.is_illicit === 1);
      
      let colorBg = '#00f59b';  // Licit emerald
      let colorBorder = 'rgba(0, 245, 155, 0.8)';
      
      if (isIllicit) {
        colorBg = '#ff2a6d';    // Illicit neon rose
        colorBorder = '#ffffff';
      } else if (n.entity_type === 'Commercial Entity') {
        colorBg = '#3a86ff';    // Commercial blue
        colorBorder = 'rgba(58, 134, 255, 0.8)';
      } else if (n.entity_type === 'Crypto Exchange') {
        colorBg = '#ffbe0b';    // Exchange yellow
        colorBorder = 'rgba(255, 190, 11, 0.8)';
      }

      const size = Math.max(12, Math.min(26, 10 + (n.in_degree + n.out_degree) * 1.5));

      return {
        id: n.id,
        label: n.label,
        shape: 'dot',
        size: size,
        color: {
          background: colorBg,
          border: colorBorder,
          highlight: { background: '#ffffff', border: '#00f0ff' },
          hover: { background: '#00f0ff', border: '#ffffff' }
        },
        borderWidth: isIllicit ? 2.5 : 1.5,
        shadow: {
          enabled: true,
          color: isIllicit ? 'rgba(255, 42, 109, 0.6)' : 'rgba(0, 245, 155, 0.3)',
          size: isIllicit ? 14 : 6,
          x: 0,
          y: 0
        },
        font: {
          color: '#8a99b5',
          size: 11,
          face: 'Outfit',
          strokeWidth: 2,
          strokeColor: '#070a12'
        },
        title: `<b>${n.label}</b> (${n.entity_type})<br>Inflow: $${(n.total_in || 0).toLocaleString()}<br>Outflow: $${(n.total_out || 0).toLocaleString()}<br>Typology: ${n.typology}<br>Risk Score: ${Math.round((n.risk_score || 0) * 100)}%`
      };
    });

    // Format edges for Vis.js
    const formattedEdges = edgesData.map((e, idx) => {
      const isIllicit = e.is_illicit === 1;
      return {
        id: `e-${idx}`,
        edgeIdx: idx,
        from: e.from,
        to: e.to,
        arrows: { to: { enabled: true, scaleFactor: 0.6 } },
        color: {
          color: isIllicit ? 'rgba(255, 42, 109, 0.75)' : 'rgba(138, 153, 181, 0.25)',
          highlight: '#00f0ff',
          hover: '#00f0ff'
        },
        width: isIllicit ? 2.5 : 1.0,
        smooth: {
          enabled: true,
          type: 'continuous',
          roundness: 0.18
        },
        title: `Click to inspect: $${(e.amount || 0).toLocaleString()} | ${e.typology || 'Transfer'}`
      };
    });

    state.nodesDataset = new vis.DataSet(formattedNodes);
    state.edgesDataset = new vis.DataSet(formattedEdges);

    const container = el.graphContainer;
    const data = { nodes: state.nodesDataset, edges: state.edgesDataset };

    const options = {
      nodes: {
        scaling: { min: 12, max: 32 },
        borderWidthSelected: 3,
        chosen: {
          node: (values, id, selected, hovering) => {
            if (selected) {
              values.borderWidth = 3;
              values.borderColor = '#00f0ff';
              values.shadow = true;
              values.shadowColor = 'rgba(0, 240, 255, 0.85)';
              values.shadowSize = 18;
            } else if (hovering) {
              values.borderWidth = 2.5;
              values.shadow = true;
              values.shadowColor = 'rgba(0, 240, 255, 0.65)';
              values.shadowSize = 14;
            }
          }
        }
      },
      edges: {
        selectionWidth: 2.5,
        smooth: {
          type: 'curvedCW',
          roundness: 0.15
        }
      },
      physics: {
        enabled: state.physicsEnabled,
        barnesHut: {
          gravitationalConstant: -2800,
          centralGravity: 0.28,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09
        },
        stabilization: {
          iterations: 100
        }
      },
      interaction: {
        hover: true,
        hoverConnectedEdges: true,
        tooltipDelay: 100,
        hideEdgesOnDrag: false,
        hideEdgesOnZoom: false,
        zoomView: true,
        dragView: true,
        navigationButtons: false
      }
    };

    if (state.network) {
      state.network.destroy();
    }

    state.network = new vis.Network(container, data, options);

    // Node & Edge click event handlers
    state.network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const selectedId = params.nodes[0];
        handleNodeSelected(selectedId);
      } else if (params.edges.length > 0) {
        handleEdgeSelected(params.edges[0]);
      } else {
        // Blank canvas clicked: smooth deselect without destroying or resetting the network!
        resetGraphHighlight();
      }
    });

    // Double click to focus and zoom in smoothly
    state.network.on('doubleClick', (params) => {
      if (params.nodes.length > 0) {
        state.network.focus(params.nodes[0], {
          scale: 1.6,
          animation: { duration: 500, easingFunction: 'easeInOutCubic' }
        });
      }
    });
  }

  // =========================================================================
  // Node Selection & Forensic Explainability (XAI)
  // =========================================================================
  async function handleNodeSelected(nodeId) {
    state.selectedNodeId = nodeId;
    setStatus(`ANALYZING EGO-NETWORK // NODE #${nodeId}`, true);

    try {
      const resp = await fetch(`/api/explain/${nodeId}`);
      const res = await resp.json();
      if (res.status === 'success') {
        renderForensicDossier(res.explanation);
        highlightEgoNetwork(res.explanation);
        // Smoothly center camera on selected node
        if (state.network) {
          state.network.focus(nodeId, {
            scale: 1.35,
            animation: { duration: 550, easingFunction: 'easeInOutCubic' }
          });
        }
      }
      setStatus(`INSPECTING ACC #${nodeId}`, false);
    } catch (err) {
      console.error('Error fetching node explanation:', err);
      setStatus('ERROR IN XAI ENGINE', false);
    }
  }

  function renderForensicDossier(xai) {
    el.dossierEmptyState.style.display = 'none';
    el.dossierContent.style.display = 'block';
    el.dossierTabs.style.display = 'flex';

    const riskPct = Math.round((xai.predicted_risk || 0) * 100);
    const isIllicit = (xai.predicted_risk || 0) >= state.currentThreshold;

    el.dossierAccountId.textContent = xai.account_id;
    el.dossierEntityType.textContent = xai.entity_name;
    el.dossierRiskNumber.textContent = `${riskPct}%`;
    el.dossierGaugeFill.style.width = `${riskPct}%`;

    // Status Badge
    if (isIllicit) {
      el.dossierStatusBadge.textContent = 'FLAGGED ILLICIT';
      el.dossierStatusBadge.className = 'badge badge-alert';
      el.dossierGaugeFill.style.background = 'var(--neon-rose)';
      el.dossierRiskNumber.style.color = 'var(--neon-rose)';
    } else {
      el.dossierStatusBadge.textContent = 'CLEARED / BENIGN';
      el.dossierStatusBadge.className = 'badge badge-safe';
      el.dossierGaugeFill.style.background = 'var(--neon-emerald)';
      el.dossierRiskNumber.style.color = 'var(--neon-emerald)';
    }

    // Typology diagnosis
    el.dossierTypologyName.textContent = xai.typology !== 'Licit' ? xai.typology : 'Normal Activity Pattern';
    el.dossierTypologyDetail.textContent = (xai.risk_factors && xai.risk_factors.length > 0)
      ? xai.risk_factors.join(' ')
      : 'No topological anomalies detected.';

    // Topological Metrics
    el.dossierInflow.textContent = `$${(xai.total_in || 0).toLocaleString()}`;
    el.dossierOutflow.textContent = `$${(xai.total_out || 0).toLocaleString()}`;
    el.dossierInDeg.textContent = xai.in_degree || 0;
    el.dossierOutDeg.textContent = xai.out_degree || 0;
    el.dossierBetweenness.textContent = (xai.betweenness || 0).toFixed(4);
    el.dossierPageRank.textContent = (xai.pagerank || 0).toFixed(4);

    // XAI Header
    const activeArch = xai.active_model || state.activeModel;
    el.dossierXaiHeader.innerHTML = `<i class="fa-solid fa-network-wired"></i> ${activeArch} Neighbor Influence`;

    // Neighbor Attention / Influence Weights List (XAI)
    el.dossierAttentionList.innerHTML = '';
    const attnEntries = Object.entries(xai.gat_attentions || {});

    if (attnEntries.length === 0) {
      el.dossierAttentionList.innerHTML = `<div class="attention-item" style="color: var(--text-dim); justify-content:center;">No direct topological neighbors</div>`;
    } else {
      attnEntries.sort((a, b) => b[1] - a[1]);
      attnEntries.slice(0, 6).forEach(([neighborId, weight]) => {
        const neighborAcc = state.rawNodes.find(n => n.id == neighborId)?.label || `ACC-${neighborId}`;
        const pct = Math.round(weight * 100);
        const item = document.createElement('div');
        item.className = 'attention-item';
        item.style.cursor = 'pointer';
        item.title = 'Click to inspect this connected account';
        item.innerHTML = `
          <span class="attn-node">${neighborAcc}</span>
          <div class="attn-bar-box">
            <div class="attn-bar" style="width: ${pct}%"></div>
            <span class="attn-val">${pct}%</span>
          </div>
        `;
        item.addEventListener('click', () => handleNodeSelected(parseInt(neighborId)));
        el.dossierAttentionList.appendChild(item);
      });
    }

    // Render Associated Transactions Ledger
    renderAccountTransactions(xai.transactions || []);
  }

  function renderAccountTransactions(transactions) {
    el.dossierTxnCount.textContent = transactions.length;
    el.accountLedgerBody.innerHTML = '';

    if (!transactions || transactions.length === 0) {
      el.accountLedgerBody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:1.5rem; color:var(--text-dim);">No transactions recorded for this account.</td></tr>';
      return;
    }

    transactions.forEach(t => {
      const isInflow = t.direction === 'INFLOW';
      const isIll = t.is_illicit === 1;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>
          <span class="flow-badge ${isInflow ? 'inflow' : 'outflow'}">
            <i class="fa-solid ${isInflow ? 'fa-arrow-down-left' : 'fa-arrow-up-right'}"></i>
            ${t.direction}
          </span>
        </td>
        <td>
          <span class="clickable-acc" style="color:var(--neon-cyan); cursor:pointer; font-weight:600;">
            ${t.counterparty}
          </span>
        </td>
        <td style="font-weight:700; color:${isIll ? 'var(--neon-rose)' : '#ffffff'};">
          $${(t.amount || 0).toLocaleString()}
        </td>
        <td style="color:var(--text-muted);">${t.payment_type || 'Transfer'}</td>
        <td style="color:var(--text-dim); font-size:0.7rem;">${t.date !== '—' ? t.date : ''} ${t.time !== '—' ? t.time : ''}</td>
        <td>
          <span style="color:${isIll ? 'var(--neon-rose)' : 'var(--text-muted)'}; font-size:0.7rem;">
            ${t.typology || 'Normal'}
          </span>
        </td>
      `;

      // Click counterparty to jump to that node
      tr.querySelector('.clickable-acc').addEventListener('click', () => {
        const targetNode = isInflow ? t.from_node : t.to_node;
        handleNodeSelected(targetNode);
      });

      el.accountLedgerBody.appendChild(tr);
    });
  }

  // =========================================================================
  // Transaction Edge Inspector
  // =========================================================================
  async function handleEdgeSelected(visEdgeId) {
    const edgeObj = state.edgesDataset.get(visEdgeId);
    if (!edgeObj) return;

    const u = edgeObj.from;
    const v = edgeObj.to;
    state.selectedEdge = edgeObj;

    try {
      const resp = await fetch(`/api/transaction/${u}/${v}`);
      const res = await resp.json();
      if (res.status === 'success') {
        renderEdgeModal(res.transaction);
      }
    } catch (err) {
      console.error('Edge inspect error:', err);
    }
  }

  function renderEdgeModal(tx) {
    const isIll = tx.is_illicit === 1;
    el.edgeModalBody.innerHTML = `
      <div class="edge-inspector-hero">
        <div class="edge-entity-box">
          <div class="edge-entity-label">Originator (Sender)</div>
          <div class="edge-entity-acc">${tx.from_account}</div>
          <div style="font-size:0.72rem; color:var(--text-muted);">${tx.sender_loc || 'Domestic Bank'}</div>
        </div>
        <div class="edge-flow-arrow">
          <div class="edge-flow-amount ${isIll ? 'alert-amount' : ''}">$${(tx.amount || 0).toLocaleString()}</div>
          <div style="font-size:0.72rem; color:var(--neon-cyan); letter-spacing:1px; margin-top:2px;">
            <i class="fa-solid fa-arrow-right-long"></i> ${tx.payment_type || 'TRANSFER'}
          </div>
        </div>
        <div class="edge-entity-box">
          <div class="edge-entity-label">Beneficiary (Receiver)</div>
          <div class="edge-entity-acc">${tx.to_account}</div>
          <div style="font-size:0.72rem; color:var(--text-muted);">${tx.receiver_loc || 'Domestic Bank'}</div>
        </div>
      </div>

      <div class="edge-features-grid">
        <div class="edge-feat-tile">
          <div class="edge-feat-lbl">Status &amp; Alert Flag</div>
          <div class="edge-feat-val" style="color:${isIll ? 'var(--neon-rose)' : 'var(--neon-emerald)'};">
            ${isIll ? '⚠️ FLAGGED ILLICIT' : '✔ LICIT / BENIGN'}
          </div>
        </div>
        <div class="edge-feat-tile">
          <div class="edge-feat-lbl">AML Typology</div>
          <div class="edge-feat-val">${tx.typology || 'Standard Transfer'}</div>
        </div>
        <div class="edge-feat-tile">
          <div class="edge-feat-lbl">Currency</div>
          <div class="edge-feat-val">${tx.currency || 'USD'}</div>
        </div>
        <div class="edge-feat-tile">
          <div class="edge-feat-lbl">Timestamp / Date</div>
          <div class="edge-feat-val">${tx.date && tx.date !== '\u2014' ? tx.date : ''} ${tx.time && tx.time !== '\u2014' ? tx.time : ''}</div>
        </div>
        <div class="edge-feat-tile">
          <div class="edge-feat-lbl">Sender Location</div>
          <div class="edge-feat-val">${tx.sender_loc || '\u2014'}</div>
        </div>
        <div class="edge-feat-tile">
          <div class="edge-feat-lbl">Receiver Location</div>
          <div class="edge-feat-val">${tx.receiver_loc || '\u2014'}</div>
        </div>
      </div>
    `;

    // Render ALL custom CSV features if present (full forensic record)
    const customFeats = tx.custom_features || {};
    const featEntries = Object.entries(customFeats).filter(([k, v]) => {
      const vs = String(v == null ? '' : v).trim();
      return vs !== '' && vs !== 'null' && vs !== 'undefined';
    });
    if (featEntries.length > 0) {
      const section = document.createElement('div');
      section.className = 'edge-custom-features';
      const titleEl = document.createElement('div');
      titleEl.className = 'edge-custom-title';
      titleEl.innerHTML = '<i class="fa-solid fa-table-columns"></i> Full CSV Feature Record &mdash; ' + featEntries.length + ' dataset columns';
      section.appendChild(titleEl);
      const grid = document.createElement('div');
      grid.className = 'edge-custom-grid';
      featEntries.forEach(function([key, val]) {
        const pair = document.createElement('div');
        pair.className = 'edge-custom-pair';
        pair.title = key + ': ' + val;
        pair.innerHTML = '<div class="edge-custom-key">' + key.replace(/_/g, ' ') + '</div><div class="edge-custom-val">' + String(val) + '</div>';
        grid.appendChild(pair);
      });
      section.appendChild(grid);
      el.edgeModalBody.appendChild(section);
    }

    el.edgeModal.style.display = 'flex';
  }

  // =========================================================================
  // Global Transaction Explorer Modal
  // =========================================================================
  function handleOpenExplorer() {
    el.explorerModal.style.display = 'flex';
    loadExplorerTransactions();
  }

  async function loadExplorerTransactions() {
    const search = el.explorerSearchInput.value.trim();
    const illicitOnly = el.explorerIllicitOnly.checked;

    el.explorerTableBody.innerHTML = '<tr><td colspan="10" class="text-center" style="padding:2rem;"><i class="fa-solid fa-spinner fa-spin"></i> Querying transactions ledger...</td></tr>';

    try {
      const url = `/api/transactions?search=${encodeURIComponent(search)}&illicit_only=${illicitOnly}&limit=120`;
      const resp = await fetch(url);
      const res = await resp.json();

      if (res.status === 'success') {
        el.explorerTxnCount.textContent = `Showing ${res.transactions.length} of ${res.total} transactions`;
        el.explorerTableBody.innerHTML = '';

        if (res.transactions.length === 0) {
          el.explorerTableBody.innerHTML = '<tr><td colspan="10" class="text-center" style="padding:2rem; color:var(--text-dim);">No transactions matched query.</td></tr>';
          return;
        }

        res.transactions.forEach(t => {
          const isIll = t.is_illicit === 1;
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="font-weight:700; color:#ffffff;">${t.from_account}</td>
            <td style="font-weight:700; color:var(--neon-cyan);">${t.to_account}</td>
            <td style="font-weight:700; color:${isIll ? 'var(--neon-rose)' : 'var(--neon-emerald)'};">$${(t.amount || 0).toLocaleString()}</td>
            <td>${t.payment_type || 'Transfer'}</td>
            <td>${t.currency || 'USD'}</td>
            <td style="font-size:0.72rem; color:var(--text-dim);">${t.date !== '—' ? t.date : ''} ${t.time !== '—' ? t.time : ''}</td>
            <td style="font-size:0.72rem; color:var(--text-muted);">${t.sender_loc} &rarr; ${t.receiver_loc}</td>
            <td>${t.typology}</td>
            <td>
              <span class="badge ${isIll ? 'badge-alert' : 'badge-safe'}" style="font-size:0.65rem;">
                ${isIll ? 'ILLICIT' : 'LICIT'}
              </span>
            </td>
            <td>
              <button class="btn btn-secondary" style="padding:0.2rem 0.5rem; font-size:0.7rem;">
                <i class="fa-solid fa-crosshairs"></i>
              </button>
            </td>
          `;

          // Crosshairs button focuses transaction on network
          tr.querySelector('button').addEventListener('click', () => {
            el.explorerModal.style.display = 'none';
            handleNodeSelected(t.from_node);
          });

          el.explorerTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error('Explorer query error:', err);
      el.explorerTableBody.innerHTML = '<tr><td colspan="10" class="text-center" style="color:var(--neon-rose);">Failed to query transactions.</td></tr>';
    }
  }

  function highlightEgoNetwork(xai) {
    if (!state.network || !state.nodesDataset) return;

    const egoNodeIds = new Set((xai.ego_nodes || []).map(n => n.id));
    const allNodeIds = state.nodesDataset.getIds();

    const nodeUpdates = [];
    allNodeIds.forEach(id => {
      if (id === xai.node_id) {
        nodeUpdates.push({
          id: id,
          color: {
            background: '#ffffff',
            border: '#00f0ff',
            highlight: { background: '#ffffff', border: '#00f0ff' }
          },
          size: 32,
          borderWidth: 3.5,
          shadow: {
            enabled: true,
            color: 'rgba(0, 240, 255, 0.95)',
            size: 22,
            x: 0,
            y: 0
          }
        });
      } else if (egoNodeIds.has(id)) {
        const rawNode = state.rawNodes.find(n => n.id === id);
        const isIllicit = (rawNode?.pred_is_illicit === 1) || (rawNode?.is_illicit === 1);
        nodeUpdates.push({
          id: id,
          color: {
            background: isIllicit ? '#ff2a6d' : '#00f59b',
            border: '#ffffff'
          },
          size: 22,
          borderWidth: 2,
          shadow: {
            enabled: true,
            color: isIllicit ? 'rgba(255, 42, 109, 0.75)' : 'rgba(0, 245, 155, 0.55)',
            size: 14,
            x: 0,
            y: 0
          }
        });
      } else {
        nodeUpdates.push({
          id: id,
          color: {
            background: 'rgba(16, 24, 43, 0.28)',
            border: 'rgba(255, 255, 255, 0.05)'
          },
          size: 10,
          borderWidth: 1,
          shadow: { enabled: false }
        });
      }
    });

    state.nodesDataset.update(nodeUpdates);

    // Smoothly highlight ego connections and dim background edges
    if (state.edgesDataset) {
      const edgeUpdates = state.rawEdges.map((e, idx) => {
        const isDirect = (e.from === xai.node_id || e.to === xai.node_id);
        const isEgoSub = (egoNodeIds.has(e.from) && egoNodeIds.has(e.to));
        if (isDirect) {
          return {
            id: `e-${idx}`,
            color: { color: '#00f0ff', highlight: '#ffffff', hover: '#00f0ff' },
            width: 3.0
          };
        } else if (isEgoSub) {
          return {
            id: `e-${idx}`,
            color: { color: 'rgba(0, 240, 255, 0.45)', highlight: '#00f0ff', hover: '#00f0ff' },
            width: 1.8
          };
        } else {
          return {
            id: `e-${idx}`,
            color: { color: 'rgba(138, 153, 181, 0.08)', highlight: '#00f0ff', hover: '#00f0ff' },
            width: 0.8
          };
        }
      });
      state.edgesDataset.update(edgeUpdates);
    }
  }

  function resetGraphHighlight() {
    if (!state.network || !state.nodesDataset) return;
    state.selectedNodeId = null;
    state.selectedEdge = null;
    el.dossierEmptyState.style.display = 'block';
    el.dossierContent.style.display = 'none';
    el.dossierTabs.style.display = 'none';

    // Unselect node in Vis.js canvas
    state.network.unselectAll();

    // Smoothly restore default node visual styles IN-PLACE without touching node positions (x, y)
    const nodeUpdates = state.rawNodes.map(n => {
      const isIllicit = n.pred_is_illicit !== undefined ? (n.pred_is_illicit === 1) : (n.is_illicit === 1);
      
      let colorBg = '#00f59b';
      let colorBorder = 'rgba(0, 245, 155, 0.8)';
      if (isIllicit) {
        colorBg = '#ff2a6d';
        colorBorder = '#ffffff';
      } else if (n.entity_type === 'Commercial Entity') {
        colorBg = '#3a86ff';
        colorBorder = 'rgba(58, 134, 255, 0.8)';
      } else if (n.entity_type === 'Crypto Exchange') {
        colorBg = '#ffbe0b';
        colorBorder = 'rgba(255, 190, 11, 0.8)';
      }

      const size = Math.max(12, Math.min(26, 10 + (n.in_degree + n.out_degree) * 1.5));

      return {
        id: n.id,
        hidden: false,
        size: size,
        color: {
          background: colorBg,
          border: colorBorder,
          highlight: { background: '#ffffff', border: '#00f0ff' },
          hover: { background: '#00f0ff', border: '#ffffff' }
        },
        borderWidth: isIllicit ? 2.5 : 1.5,
        shadow: {
          enabled: true,
          color: isIllicit ? 'rgba(255, 42, 109, 0.6)' : 'rgba(0, 245, 155, 0.3)',
          size: isIllicit ? 14 : 6,
          x: 0,
          y: 0
        },
        opacity: 1.0
      };
    });

    state.nodesDataset.update(nodeUpdates);

    // Smoothly restore edge styles in-place
    if (state.edgesDataset) {
      const edgeUpdates = state.rawEdges.map((e, idx) => {
        const isIllicit = e.is_illicit === 1;
        return {
          id: `e-${idx}`,
          color: {
            color: isIllicit ? 'rgba(255, 42, 109, 0.75)' : 'rgba(138, 153, 181, 0.25)',
            highlight: '#00f0ff',
            hover: '#00f0ff'
          },
          width: isIllicit ? 2.5 : 1.0,
          opacity: 1.0
        };
      });
      state.edgesDataset.update(edgeUpdates);
    }

    el.typologyFilterContainer.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    const allBtn = el.typologyFilterContainer.querySelector('.filter-btn[data-typology="all"]');
    if (allBtn) allBtn.classList.add('active');

    setStatus(`ACTIVE ARCH: ${state.activeModel}`, false);
  }

  // =========================================================================
  // Typology Filtering
  // =========================================================================
  function applyTypologyFilter(typology) {
    if (!state.nodesDataset) return;

    const allNodeIds = state.nodesDataset.getIds();
    const nodeUpdates = [];

    allNodeIds.forEach(id => {
      const rawNode = state.rawNodes.find(n => n.id === id);
      if (!rawNode) return;

      let match = false;
      if (typology === 'all') {
        match = true;
      } else if (typology === 'Illicit') {
        match = (rawNode.pred_is_illicit === 1) || (rawNode.is_illicit === 1);
      } else {
        match = (rawNode.typology || '').toLowerCase().includes(typology.toLowerCase());
      }

      if (match) {
        const isIllicit = (rawNode.pred_is_illicit === 1) || (rawNode.is_illicit === 1);
        nodeUpdates.push({
          id: id,
          hidden: false,
          color: {
            background: isIllicit ? '#ff2a6d' : '#00f59b',
            border: isIllicit ? '#ffffff' : 'rgba(0,245,155,0.8)'
          },
          size: isIllicit ? 24 : 16
        });
      } else {
        nodeUpdates.push({
          id: id,
          color: { background: 'rgba(20, 30, 50, 0.2)', border: 'rgba(255,255,255,0.04)' },
          size: 8
        });
      }
    });

    state.nodesDataset.update(nodeUpdates);
    if (state.network) state.network.fit({ animation: { duration: 500 } });
  }

  // =========================================================================
  // GNN Model Training Loop
  // =========================================================================
  function initTrainingChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    state.trainingChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Train Loss (Focal)',
            data: [],
            borderColor: '#ff2a6d',
            backgroundColor: 'rgba(255, 42, 109, 0.1)',
            tension: 0.35,
            fill: true,
            yAxisID: 'y'
          },
          {
            label: 'Validation ROC-AUC',
            data: [],
            borderColor: '#00f0ff',
            backgroundColor: 'transparent',
            tension: 0.35,
            borderDash: [5, 5],
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8a99b5', font: { family: 'JetBrains Mono', size: 10 } } },
          y: { type: 'linear', position: 'left', ticks: { color: '#ff2a6d' } },
          y1: { type: 'linear', position: 'right', min: 0.5, max: 1.0, ticks: { color: '#00f0ff' }, grid: { drawOnChartArea: false } }
        },
        plugins: {
          legend: { labels: { color: '#8a99b5', font: { family: 'Outfit' } } }
        }
      }
    });
  }

  async function handleTrainModel() {
    const modelName = state.activeModel;
    const epochs = parseInt(el.epochsInput.value);
    const lr = parseFloat(el.lrInput.value);
    const lossType = el.lossTypeSelect.value;

    setStatus(`TRAINING ${modelName} (${epochs} EPOCHS)...`, true);
    el.btnTrainModel.disabled = true;
    el.btnTrainModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Optimizing Graph Weights...';

    try {
      const resp = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          epochs: epochs,
          lr: lr,
          loss_type: lossType
        })
      });

      const res = await resp.json();
      if (res.status === 'success') {
        const result = res.result;
        updateTrainingTelemetry(result.history);
        updateMetricsDisplay(result.metrics);
        el.bottomDrawer.classList.add('open');
        setStatus(`TRAINED ${modelName} IN ${result.train_time_sec}s`, false);
      } else {
        setStatus('TRAINING FAILED', false);
      }
    } catch (err) {
      console.error('Training error:', err);
      setStatus('TRAINING ERROR', false);
    } finally {
      el.btnTrainModel.disabled = false;
      el.btnTrainModel.innerHTML = '<i class="fa-solid fa-bolt"></i> Train Model';
    }
  }

  function updateTrainingTelemetry(history) {
    if (!state.trainingChart || !history) return;

    const epochs = history.map(h => `E${h.epoch}`);
    const losses = history.map(h => h.train_loss);
    const aucs = history.map(h => h.val_auc);

    state.trainingChart.data.labels = epochs;
    state.trainingChart.data.datasets[0].data = losses;
    state.trainingChart.data.datasets[1].data = aucs;
    state.trainingChart.update();
  }

  function updateMetricsDisplay(metrics) {
    if (!metrics) return;

    el.kpiAuc.textContent = (metrics.roc_auc || 0).toFixed(3);

    const cm = metrics.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 };
    el.cmTN.textContent = cm.tn;
    el.cmFP.textContent = cm.fp;
    el.cmFN.textContent = cm.fn;
    el.cmTP.textContent = cm.tp;

    const prec = Math.round((metrics.precision || 0) * 1000) / 10;
    const rec = Math.round((metrics.recall || 0) * 1000) / 10;
    const f1 = Math.round((metrics.f1_score || 0) * 1000) / 10;

    el.statPrec.textContent = `${prec}%`;
    el.statRecall.textContent = `${rec}%`;
    el.statF1.textContent = `${f1}%`;
    el.statPrAuc.textContent = (metrics.pr_auc || 0).toFixed(3);

    el.drawerSummaryMetrics.textContent = `Precision: ${prec}% | Recall: ${rec}% | F1: ${f1}% | ROC-AUC: ${(metrics.roc_auc || 0).toFixed(3)}`;
  }

  // =========================================================================
  // Threshold Slider Tuning & Dynamic Model Evaluation
  // =========================================================================
  async function updatePredictionThreshold(threshold) {
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threshold: threshold,
          model_name: state.activeModel
        })
      });

      const res = await resp.json();
      if (res.status === 'success') {
        updateMetricsDisplay(res.metrics);
        if (res.nodes) {
          state.rawNodes = res.nodes;
          const updates = res.nodes.map(n => ({
            id: n.id,
            color: {
              background: n.pred_is_illicit === 1 ? '#ff2a6d' : '#00f59b',
              border: n.pred_is_illicit === 1 ? '#ffffff' : 'rgba(0, 245, 155, 0.8)'
            }
          }));
          if (state.nodesDataset) state.nodesDataset.update(updates);
        }
      }
    } catch (err) {
      console.error('Threshold prediction update error:', err);
    }
  }

  // =========================================================================
  // Multi-Architecture Benchmark Arena
  // =========================================================================
  async function handleOpenBenchmark() {
    el.benchmarkModal.style.display = 'flex';
    el.benchmarkTableBody.innerHTML = '<tr><td colspan="8" class="text-center" style="padding:2rem;"><i class="fa-solid fa-spinner fa-spin"></i> Benchmarking GCN, GAT, GraphSAGE &amp; Baseline MLP on test split...</td></tr>';

    try {
      const resp = await fetch('/api/benchmark');
      const res = await resp.json();
      if (res.status === 'success') {
        renderBenchmarkResults(res.benchmarks);
      }
    } catch (err) {
      console.error('Benchmark arena error:', err);
      el.benchmarkTableBody.innerHTML = '<tr><td colspan="8" class="text-center" style="color:var(--neon-rose);">Failed to run benchmark suite.</td></tr>';
    }
  }

  function renderBenchmarkResults(benchmarks) {
    el.benchmarkTableBody.innerHTML = '';

    const modelLabels = [];
    const aucData = [];
    const f1Data = [];

    benchmarks.forEach(m => {
      const isGnn = m.model_name !== 'Baseline (MLP)';
      const mechanism = m.model_name === 'GAT' ? 'Multi-Head Attention'
        : m.model_name === 'GCN' ? 'Normalized Graph Laplacian'
        : m.model_name === 'GraphSAGE' ? 'Sample & Mean Aggregation'
        : 'Tabular MLP (No Graph)';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:700; color:${isGnn ? '#ffffff' : '#8a99b5'};">${m.model_name}</td>
        <td style="font-size:0.74rem; color:var(--neon-cyan);">${mechanism}</td>
        <td style="color:var(--neon-emerald); font-weight:700;">${(m.roc_auc || 0).toFixed(4)}</td>
        <td>${(m.pr_auc || 0).toFixed(4)}</td>
        <td>${Math.round((m.precision || 0) * 100)}%</td>
        <td>${Math.round((m.recall || 0) * 100)}%</td>
        <td style="font-weight:700;">${Math.round((m.f1_score || 0) * 100)}%</td>
        <td>${m.train_time_sec || 1.1}s</td>
      `;
      el.benchmarkTableBody.appendChild(tr);

      modelLabels.push(m.model_name);
      aucData.push(m.roc_auc);
      f1Data.push(m.f1_score);
    });

    // Render Benchmark Chart
    const ctx = document.getElementById('benchmarkChart').getContext('2d');
    if (state.benchmarkChart) state.benchmarkChart.destroy();

    state.benchmarkChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: modelLabels,
        datasets: [
          {
            label: 'ROC-AUC',
            data: aucData,
            backgroundColor: 'rgba(0, 240, 255, 0.7)',
            borderColor: '#00f0ff',
            borderWidth: 1
          },
          {
            label: 'F1-Score',
            data: f1Data,
            backgroundColor: 'rgba(0, 245, 155, 0.7)',
            borderColor: '#00f59b',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#ffffff', font: { family: 'Outfit', weight: '600' } } },
          y: { min: 0.4, max: 1.0, ticks: { color: '#8a99b5', font: { family: 'JetBrains Mono' } } }
        },
        plugins: {
          legend: { labels: { color: '#8a99b5', font: { family: 'Outfit' } } }
        }
      }
    });
  }

  // =========================================================================
  // FinCEN SAR Regulatory Generation
  // =========================================================================
  async function handleExportSAR() {
    if (state.selectedNodeId === null) return;

    setStatus('GENERATING SAR DOSSIER...', true);
    try {
      const resp = await fetch(`/api/sar/${state.selectedNodeId}`);
      const res = await resp.json();
      if (res.status === 'success') {
        state.currentSarData = res.sar;
        el.sarJsonContent.textContent = JSON.stringify(res.sar, null, 2);
        el.sarModal.style.display = 'flex';
      }
      setStatus('ENGINE READY', false);
    } catch (err) {
      console.error('Error generating SAR:', err);
      setStatus('SAR GENERATION FAILED', false);
    }
  }

  // =========================================================================
  // 1-Click Instant SAML-D.csv Loader
  // =========================================================================
  async function handleLoadLocalSaml() {
    const sampleSize = el.samlSampleSize.value;
    setStatus('STREAMING SAML-D.CSV (1 GB)...', true);
    document.getElementById('uploadStep1').style.display = 'none';
    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('uploadProgressLabel').textContent = `Streaming and extracting ${sampleSize} transactions from SAML-D.csv...`;
    document.getElementById('uploadProgressBar').style.width = '50%';

    try {
      const resp = await fetch('/api/load-local-saml', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_nodes: parseInt(sampleSize) })
      });
      const res = await resp.json();
      document.getElementById('uploadProgressBar').style.width = '100%';

      if (res.status === 'success') {
        closeUploadModal();
        await loadGraphData();
        const info = res.summary.sample_info || {};
        setStatus(`SAML-D LOADED (${res.summary.num_nodes} accounts, ${res.summary.num_edges} txns)`, false);
      } else {
        alert('Failed to load SAML-D: ' + res.message);
        openUploadModal();
        setStatus('SAML-D LOAD FAILED', false);
      }
    } catch (err) {
      console.error('Error loading SAML-D:', err);
      alert('Network or parsing error while loading SAML-D.csv.');
      openUploadModal();
      setStatus('SAML-D ERROR', false);
    }
  }

  // =========================================================================
  // Upload Dataset Modal (Custom CSV)
  // =========================================================================
  let uploadFile = null;
  let detectedHeaders = [];
  let detectedMapping = {};

  const ROLE_SPECS = {
    sender: {
      label: 'Sender Account ID',
      cat: 'cat-topo',
      catName: 'Topology',
      catIcon: 'fa-diagram-project',
      desc: 'Source node for graph adjacency matrix'
    },
    receiver: {
      label: 'Receiver Account ID',
      cat: 'cat-topo',
      catName: 'Topology',
      catIcon: 'fa-diagram-project',
      desc: 'Target node for graph adjacency matrix'
    },
    amount: {
      label: 'Transaction Amount',
      cat: 'cat-flow',
      catName: 'Flow Tensor',
      catIcon: 'fa-money-bill-wave',
      desc: 'Continuous numerical value feature'
    },
    label: {
      label: 'Laundering Label (0/1)',
      cat: 'cat-target',
      catName: 'Supervised Target',
      catIcon: 'fa-crosshairs',
      desc: 'Ground-truth binary label for GNN loss'
    },
    typology: {
      label: 'Laundering Typology',
      cat: 'cat-meta',
      catName: 'Forensic Meta',
      catIcon: 'fa-fingerprint',
      desc: 'Structuring, cycle, layering classification'
    },
    currency: {
      label: 'Currency',
      cat: 'cat-flow',
      catName: 'Flow Tensor',
      catIcon: 'fa-coins',
      desc: 'Payment currency (USD, EUR, BTC, etc.)'
    },
    payment_type: {
      label: 'Payment Method',
      cat: 'cat-flow',
      catName: 'Flow Tensor',
      catIcon: 'fa-credit-card',
      desc: 'Wire, ACH, Cash, Crypto transfer method'
    },
    sender_location: {
      label: 'Sender Location',
      cat: 'cat-meta',
      catName: 'Forensic Meta',
      catIcon: 'fa-location-dot',
      desc: 'Originating jurisdiction or branch'
    },
    receiver_location: {
      label: 'Receiver Location',
      cat: 'cat-meta',
      catName: 'Forensic Meta',
      catIcon: 'fa-location-crosshairs',
      desc: 'Beneficiary jurisdiction or branch'
    },
    date: {
      label: 'Date',
      cat: 'cat-meta',
      catName: 'Temporal',
      catIcon: 'fa-calendar',
      desc: 'Transaction timestamp for sequence flow'
    },
    time: {
      label: 'Time',
      cat: 'cat-meta',
      catName: 'Temporal',
      catIcon: 'fa-clock',
      desc: 'Burst & velocity rate feature'
    },
  };

  function openUploadModal() {
    uploadFile = null;
    detectedHeaders = [];
    detectedMapping = {};
    document.getElementById('uploadStep1').style.display = 'block';
    document.getElementById('uploadStep2').style.display = 'none';
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('uploadSampleWarning').style.display = 'none';
    document.getElementById('btnImportDataset').disabled = true;
    document.getElementById('csvFileInput').value = '';
    document.getElementById('dropZone').classList.remove('drag-over');
    document.getElementById('uploadModal').style.display = 'flex';
  }

  function closeUploadModal() {
    document.getElementById('uploadModal').style.display = 'none';
  }

  async function handleFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    uploadFile = file;

    setStatus('DETECTING CSV HEADERS...', true);
    document.getElementById('uploadStep1').style.display = 'none';
    document.getElementById('uploadStep2').style.display = 'none';
    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('uploadProgressLabel').textContent = 'Reading CSV headers, feature roles & sample preview…';
    document.getElementById('uploadProgressBar').style.width = '35%';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch('/api/upload-csv/columns', { method: 'POST', body: formData });
      const res = await resp.json();

      document.getElementById('uploadProgress').style.display = 'none';

      if (res.status !== 'success') {
        alert('Could not read file: ' + res.message);
        setStatus('ENGINE READY', false);
        openUploadModal();
        return;
      }

      detectedHeaders = res.headers || [];
      detectedMapping = res.detected_mapping || {};

      const sizeMB = (file.size / 1048576).toFixed(1);
      document.getElementById('uploadFileInfo').innerHTML = `
        <div class="upload-file-pill">
          <i class="fa-solid fa-file-csv"></i>
          <strong>${file.name}</strong>
          <span>${sizeMB} MB</span>
          <span>${detectedHeaders.length} columns detected</span>
        </div>
      `;

      const colCountBadge = document.getElementById('detectedColCount');
      if (colCountBadge) {
        colCountBadge.textContent = `${detectedHeaders.length} Columns Detected`;
      }

      // Build enriched column mapping table with neural categories
      const tbody = document.getElementById('colMapBody');
      tbody.innerHTML = '';
      Object.entries(ROLE_SPECS).forEach(([role, spec]) => {
        const detected = detectedMapping[role] || '';
        const isRequired = ['sender','receiver','amount'].includes(role);
        const statusIcon = detected
          ? '<i class="fa-solid fa-circle-check" style="color:var(--neon-emerald)"></i>'
          : (isRequired ? '<i class="fa-solid fa-circle-xmark" style="color:var(--neon-rose)"></i>'
                        : '<i class="fa-solid fa-circle-minus" style="color:var(--text-dim)"></i>');

        const options = ['<option value="">(not mapped)</option>']
          .concat(detectedHeaders.map(h =>
            `<option value="${h}" ${h === detected ? 'selected' : ''}>${h}</option>`
          ));

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>
            <div style="font-weight:700; color:#ffffff;">${statusIcon} ${spec.label}${isRequired ? ' <span style="color:var(--neon-rose)">*</span>' : ''}</div>
            <div style="font-size:0.68rem; color:var(--text-dim); margin-top:2px;">${spec.desc}</div>
          </td>
          <td>
            <span class="feat-category-badge ${spec.cat}"><i class="fa-solid ${spec.catIcon}"></i> ${spec.catName}</span>
          </td>
          <td><code style="color:${detected ? 'var(--neon-cyan)' : 'var(--text-dim)'}; font-size:0.8rem;">${detected || '—'}</code></td>
          <td><select class="col-map-select form-select select-compact" data-role="${role}">${options.join('')}</select></td>
        `;
        tbody.appendChild(tr);
      });

      // Render Live CSV Sample Data Preview Table
      const sampleHead = document.getElementById('samplePreviewHead');
      const sampleBody = document.getElementById('samplePreviewBody');
      const sampleRows = res.sample_rows || [];
      const rowBadge = document.getElementById('previewRowBadge');
      if (rowBadge) rowBadge.textContent = `${sampleRows.length} sample rows`;

      if (sampleHead && sampleBody) {
        sampleHead.innerHTML = '';
        sampleBody.innerHTML = '';

        if (sampleRows.length > 0) {
          // Headers
          const trHead = document.createElement('tr');
          detectedHeaders.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            trHead.appendChild(th);
          });
          sampleHead.appendChild(trHead);

          // Data rows
          sampleRows.slice(0, 4).forEach(row => {
            const tr = document.createElement('tr');
            detectedHeaders.forEach(h => {
              const td = document.createElement('td');
              const val = row[h] != null ? String(row[h]) : '—';
              td.textContent = val.length > 28 ? val.substring(0, 28) + '…' : val;
              td.title = `${h}: ${val}`;
              tr.appendChild(td);
            });
            sampleBody.appendChild(tr);
          });
        } else {
          sampleBody.innerHTML = '<tr><td colspan="10" style="padding:1rem; text-align:center; color:var(--text-dim);">No sample rows available.</td></tr>';
        }
      }

      // Show sampling alert if large
      const warn = document.getElementById('uploadSampleWarning');
      if (file.size > 5 * 1048576) {
        warn.style.display = 'flex';
        document.getElementById('uploadWarningText').textContent =
          `Large dataset detected (${sizeMB} MB). The engine will use stratified streaming to extract graph connections without browser freezing.`;
      } else {
        warn.style.display = 'none';
      }

      document.getElementById('uploadStep2').style.display = 'block';
      document.getElementById('btnImportDataset').disabled = false;
      setStatus('FEATURES DETECTED', false);

    } catch (err) {
      console.error('Column detect error:', err);
      alert('Failed to read file headers.');
      setStatus('ENGINE READY', false);
      openUploadModal();
    }
  }

  async function handleImportDataset() {
    // Collect user overrides from dropdowns
    const overrides = {};
    document.querySelectorAll('.col-map-select').forEach(sel => {
      if (sel.value) overrides[sel.dataset.role] = sel.value;
    });

    if (!overrides.sender && !detectedMapping.sender) {
      alert('Sender Account column is required. Please select it in the table above.');
      return;
    }
    if (!overrides.receiver && !detectedMapping.receiver) {
      alert('Receiver Account column is required. Please select it in the table above.');
      return;
    }

    const maxNodes = el.customSampleSize ? el.customSampleSize.value : '2500';

    document.getElementById('uploadStep2').style.display = 'none';
    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('btnImportDataset').disabled = true;

    let pct = 15;
    const progressBar = document.getElementById('uploadProgressBar');
    const progressLabel = document.getElementById('uploadProgressLabel');
    progressBar.style.width = pct + '%';
    progressLabel.textContent = 'Parsing transactions & constructing graph connections…';
    const ticker = setInterval(() => {
      if (pct < 85) { pct += 8; progressBar.style.width = pct + '%'; }
      if (pct >= 40) progressLabel.textContent = 'Extracting network topology & node metrics…';
      if (pct >= 65) progressLabel.textContent = 'Optimizing GNN feature tensor matrix…';
    }, 400);

    setStatus('IMPORTING DATASET...', true);

    const formData = new FormData();
    if (uploadFile) formData.append('file', uploadFile);
    formData.append('col_map', JSON.stringify(overrides));
    formData.append('max_nodes', maxNodes);

    try {
      const resp = await fetch('/api/upload-csv', { method: 'POST', body: formData });
      clearInterval(ticker);
      progressBar.style.width = '100%';
      progressLabel.textContent = 'Rendering interactive force graph…';

      const res = await resp.json();
      if (res.status === 'success') {
        closeUploadModal();
        await loadGraphData();
        const info = res.summary.sample_info || {};
        setStatus(`DATASET IMPORTED — ${(res.summary.num_nodes || 0).toLocaleString()} accounts, ${(res.summary.num_edges || 0).toLocaleString()} txns`, false);
      } else {
        clearInterval(ticker);
        document.getElementById('uploadStep2').style.display = 'block';
        document.getElementById('uploadProgress').style.display = 'none';
        document.getElementById('btnImportDataset').disabled = false;
        alert('Import failed: ' + res.message);
        setStatus('IMPORT FAILED', false);
      }
    } catch (err) {
      clearInterval(ticker);
      console.error('Import error:', err);
      document.getElementById('uploadStep2').style.display = 'block';
      document.getElementById('uploadProgress').style.display = 'none';
      document.getElementById('btnImportDataset').disabled = false;
      alert('Network error during import.');
      setStatus('IMPORT ERROR', false);
    }
  }

  // =========================================================================
  // Network Regeneration
  // =========================================================================
  async function handleRegenerateGraph() {
    setStatus('SYNTHESIZING BENCHMARK GRAPH...', true);
    try {
      const resp = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_nodes: 520, illicit_ratio: 0.08 })
      });
      const res = await resp.json();
      if (res.status === 'success') {
        await loadGraphData();
        setStatus('BENCHMARK GRAPH SYNTHESIZED', false);
      }
    } catch (err) {
      console.error('Regenerate error:', err);
      setStatus('REGENERATE FAILED', false);
    }
  }

  // Utilities
  function setStatus(text, isBusy) {
    el.systemStatus.textContent = text;
    const dot = document.querySelector('.pulse-dot');
    if (dot) {
      dot.style.backgroundColor = isBusy ? '#00f0ff' : '#00f59b';
      dot.style.boxShadow = isBusy ? '0 0 8px #00f0ff' : '0 0 8px #00f59b';
    }
  }

  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }
});
