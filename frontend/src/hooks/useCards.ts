import { useState, useCallback } from 'react';
import { cardsApi } from '@/services/api';
import { Card, CardSearchParams } from '@/types';
import { useQuery } from '@tanstack/react-query';

export function useCardSearch() {
  const [searchParams, setSearchParams] = useState<CardSearchParams>({
    limit: 20,
    offset: 0,
    standard_only: true,
  });

  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['cards', searchParams],
    queryFn: async () => {
      const response = await cardsApi.search(searchParams);
      return response.data;
    },
    enabled: !!searchParams.q || !!searchParams.colors?.length || !!searchParams.card_type,
  });

  const search = useCallback((params: Partial<CardSearchParams>) => {
    setSearchParams((prev) => ({
      ...prev,
      ...params,
      offset: 0, // Reset pagination on new search
    }));
  }, []);

  const loadMore = useCallback(() => {
    setSearchParams((prev) => ({
      ...prev,
      offset: (prev.offset || 0) + (prev.limit || 20),
    }));
  }, []);

  return {
    cards: data || [],
    total: data?.length || 0,
    isLoading,
    error,
    search,
    loadMore,
    refetch,
  };
}

export function useCard(cardId: string | undefined) {
  return useQuery({
    queryKey: ['card', cardId],
    queryFn: async () => {
      if (!cardId) throw new Error('Card ID required');
      const response = await cardsApi.getById(cardId);
      return response.data as Card;
    },
    enabled: !!cardId,
  });
}

export function useCardByName(cardName: string | undefined) {
  return useQuery({
    queryKey: ['card-by-name', cardName],
    queryFn: async () => {
      if (!cardName) throw new Error('Card name required');
      const response = await cardsApi.getByName(cardName);
      return response.data as Card;
    },
    enabled: !!cardName,
  });
}

export function useSemanticSearch() {
  const [results, setResults] = useState<Card[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const search = useCallback(async (query: string, limit = 10) => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setIsLoading(true);
    try {
      const response = await cardsApi.semanticSearch(query, limit);
      setResults(response.data);
    } catch (error) {
      console.error('Semantic search error:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    results,
    isLoading,
    search,
  };
}
